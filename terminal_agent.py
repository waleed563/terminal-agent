"""
Terminal Agent
==============
A small terminal agent built with LangGraph, inspired by Claude Code and Codex.

It takes a task in plain English, then loops:  think -> act -> think -> ...
until the task is done. It can run shell commands and read/write files.

Main features:
- Sub-agent (explore_agent) for big search jobs, so the main chat stays clean
- Compaction: summarizes old messages when the chat gets too long
- Checkpointing: auto-saves after every step (SQLite), so it can resume after a crash
- LangSmith tracing: every step is recorded so you can watch and debug it
- Safety: command blocklist, output truncation, git auto-commit on file changes
- 3 ways to run commands: your machine, a Docker sandbox, or Terminal-Bench's terminal
"""

import os
import time
import base64
import shlex
import subprocess
import sqlite3
from typing import TypedDict, List

from dotenv import load_dotenv
load_dotenv()  # reads .env and loads ANTHROPIC_API_KEY / OPENAI_API_KEY / LANGCHAIN_API_KEY

# LangSmith tracing - once these are set, every run automatically shows up
# at smith.langchain.com with full reasoning traces, no other code needed.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "terminal-agent"

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================
# 1. STATE - the shared notebook every node reads/writes
# ============================================================

class AgentState(TypedDict):
    messages: List
    task: str
    step_count: int
    done: bool
    exploration_result: str  # holds the subagent's clean summary, if used


# FIX 4: raised from 15 to 40. Real tasks (install, configure, test)
# need many steps - 15 was stopping the agent too early.
MAX_STEPS = 40             # safety limit for the main loop
MAX_SUB_STEPS = 10         # safety limit for explore_agent's internal loop
COMPACTION_THRESHOLD = 20  # if messages list grows past this, compact it

# FIX 3: raised from 60 seconds to 3 minutes. Installs like
# "pip install jupyter" were being cut off halfway at 60 seconds.
# Kept BELOW Terminal-Bench's default 6-minute limit per task, so if one
# command gets stuck, the agent still has time to press Ctrl+C and recover.
COMMAND_TIMEOUT_SEC = 180

# BUG FIX 3 (time budget): a clock time (from time.time()) when the agent
# must stop. None = no limit (normal runs on your machine). The
# Terminal-Bench adapter sets it from each task's own time limit, so the
# agent wraps up BEFORE Terminal-Bench pulls the plug.
DEADLINE = None
WRAP_UP_SECONDS = 60   # in the last minute, tell the agent to finish up


def seconds_left():
    """Seconds until DEADLINE, or None if there is no deadline."""
    if DEADLINE is None:
        return None
    return DEADLINE - time.time()


# ============================================================
# 2. SYSTEM PROMPT - the agent's "job description"
# ============================================================

# FIX 1: before this, the agent had NO instructions, so the model acted
# like a chatbot - it explained solutions in text and stopped.
# This prompt tells it to act like a worker: do, check, retry.
SYSTEM_PROMPT = """You are an autonomous terminal agent working on a real Linux machine.
You may be running as root inside a sandboxed container.

Rules:
1. DO the task using your tools. Never just explain how to do it, and never
   paste code in your reply. Create files with the write tool and run
   commands with the bash tool.
2. Work step by step: first look around (ls, cat, check versions), then act,
   then check the result.
3. If a command fails, read the error and try a different approach.
   Do not give up after one error. Do not ask the user for help -
   nobody will answer. You must finish the task on your own.
4. To read a web page or download a file, use bash with curl, wget or python.
   The read tool only works for local files, not URLs.
5. Programs that keep running (servers, watchers) must be started in the
   background so the terminal stays free, for example:
   nohup <command> > /tmp/app.log 2>&1 &
   Then check that it is really running.
6. Installing software is allowed (apt-get, pip). Always add -y and use
   DEBIAN_FRONTEND=noninteractive so commands never stop to ask questions.
7. Before you finish, verify the task is really done: run the program,
   cat the files, run the tests, or curl the server.
   Only then reply with a short summary and no tool call.
"""


# ============================================================
# 3. HELPERS - safety + truncation
# ============================================================

BLOCKED = ["rm ", "sudo", "shutdown", "mkfs", "dd ", ">:"]


def is_safe(command: str) -> bool:
    """Returns False if the command contains any blocked word."""
    return not any(bad in command for bad in BLOCKED)


def truncate(text: str, max_lines: int = 2000) -> str:
    """Cuts very long output down to max_lines, so it doesn't flood the chat."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n...[TRUNCATED]"
    return text


def git_commit(message: str):
    """Saves every file change in git, so any change can be undone."""
    # fails silently if not in a git repo - fine for early testing
    subprocess.run(["git", "add", "-A"], capture_output=True)
    subprocess.run(["git", "commit", "-m", message], capture_output=True)


# ------------------------------------------------------------
# Docker sandboxing - when enabled, bash commands run INSIDE a
# throwaway container instead of directly on your machine.
# Set to False while developing (faster), True before any real
# deployment or benchmark run.
# ------------------------------------------------------------
USE_DOCKER_SANDBOX = False
SANDBOX_IMAGE = "python:3.11-slim"


def run_in_sandbox(command: str, workdir: str = ".") -> str:
    """Runs a command inside a throwaway Docker container."""
    abs_workdir = os.path.abspath(workdir)
    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",               # no internet access inside the box
        "--memory", "256m",                # cap memory so it can't hog the host
        "-v", f"{abs_workdir}:/workspace", # only this folder is visible inside
        "-w", "/workspace",
        SANDBOX_IMAGE,
        "bash", "-c", command,
    ]
    result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr


def run_on_host(command: str) -> str:
    """Runs a command directly on your own machine."""
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=15
    )
    return result.stdout + result.stderr


# ------------------------------------------------------------
# Terminal-Bench integration - when running under their harness,
# commands need to go through their TmuxSession instead of running
# locally or in our own Docker sandbox. This stays None for every
# normal run (CLI, testing) - only the Terminal-Bench adapter sets it.
# ------------------------------------------------------------
ACTIVE_TMUX_SESSION = None


def run_via_tmux(command: str, session) -> str:
    """Types a command into Terminal-Bench's terminal and returns what's on screen."""
    # 1. Clear the screen first, so the captured output only shows
    #    THIS command's result, not everything typed before it.
    session.send_keys(["clear", "Enter"])

    # Never let one command run past the task's deadline.
    limit = COMMAND_TIMEOUT_SEC
    left = seconds_left()
    if left is not None:
        limit = max(10, min(limit, left - 15))

    # 2. Send the real command. block=True makes Terminal-Bench add
    #    "; tmux wait -S done" so we wait until the command finishes.
    try:
        session.send_keys([command, "Enter"], block=True,
                          max_timeout_sec=limit)
    except TimeoutError:
        # FIX 3 (part 2): the command is still running. Press Ctrl+C to
        # stop it, so the NEXT command doesn't land in a busy terminal.
        output = session.capture_pane()
        session.send_keys(["C-c"])
        return (output + f"\n[STOPPED: still running after {int(limit)}s, "
                "pressed Ctrl+C. If this is a server, start it in the background "
                "with: nohup <command> > /tmp/app.log 2>&1 &]")

    # 3. Read back what's on the terminal screen now.
    return session.capture_pane()


def b64(text: str) -> str:
    """Encodes text as base64 so it can travel through the terminal safely."""
    # No problems with quotes, $ signs, or newlines inside the content.
    return base64.b64encode(text.encode()).decode()


# ============================================================
# 4. TOOLS - what the agent is allowed to do
# ============================================================

@tool
def bash(command: str) -> str:
    """Runs a shell command and returns its output."""
    if ACTIVE_TMUX_SESSION is not None:
        # FIX 2: inside Terminal-Bench, every task gets its own throwaway
        # container, so the container itself is the safety box. Blocking
        # words like "sudo" there only stopped normal work (installing nginx).
        # The blocklist still protects you on your own machine below.
        output = run_via_tmux(command, ACTIVE_TMUX_SESSION)
        return truncate(output)

    if not is_safe(command):
        return "BLOCKED: this command looked dangerous, refused to run it."

    if USE_DOCKER_SANDBOX:
        output = run_in_sandbox(command)
    else:
        output = run_on_host(command)

    return truncate(output)


@tool
def read(file_path: str) -> str:
    """Reads and returns the contents of a LOCAL file (not a URL)."""
    if ACTIVE_TMUX_SESSION is not None:
        # Under Terminal-Bench the file lives INSIDE the container,
        # so read it there with cat, not with Python on the host.
        return truncate(run_via_tmux(f"cat {shlex.quote(file_path)}", ACTIVE_TMUX_SESSION))

    if not os.path.exists(file_path):
        return f"ERROR: file not found: {file_path}"
    with open(file_path, "r") as f:
        content = f.read()
    return truncate(content)


@tool
def write(file_path: str, content: str) -> str:
    """Creates or overwrites a file with new content."""
    if ACTIVE_TMUX_SESSION is not None:
        # Write the file INSIDE the container. Content goes through
        # base64 so special characters and newlines arrive exactly.
        cmd = f"echo {b64(content)} | base64 -d > {shlex.quote(file_path)}"
        run_via_tmux(cmd, ACTIVE_TMUX_SESSION)
        return "success"

    with open(file_path, "w") as f:
        f.write(content)
    git_commit(f"agent wrote {file_path}")
    return "success"


@tool
def edit(file_path: str, old_text: str, new_text: str) -> str:
    """Replaces an exact block of old_text with new_text in a file."""
    if ACTIVE_TMUX_SESSION is not None:
        # Exact-match editing through a terminal screen isn't reliable,
        # so tell the model to use write or bash instead.
        return ("FAILED: edit is not available in this environment. "
                "Use write to rewrite the whole file, or bash.")

    if not os.path.exists(file_path):
        return f"ERROR: file not found: {file_path}"
    with open(file_path, "r") as f:
        content = f.read()
    if old_text not in content:
        return "FAILED: old_text not found exactly, no changes made."
    content = content.replace(old_text, new_text, 1)
    with open(file_path, "w") as f:
        f.write(content)
    git_commit(f"agent edited {file_path}")
    return "success"


@tool
def ls(folder_path: str = ".") -> str:
    """Lists files and folders inside a directory."""
    if ACTIVE_TMUX_SESSION is not None:
        return truncate(run_via_tmux(f"ls -la {shlex.quote(folder_path)}", ACTIVE_TMUX_SESSION))

    if not os.path.exists(folder_path):
        return f"ERROR: folder not found: {folder_path}"
    return truncate("\n".join(os.listdir(folder_path)))


@tool
def explore(question: str) -> str:
    """Use this ONLY for open-ended, multi-step investigation tasks
    (e.g. 'find every place X is handled', 'understand how Y works').
    Do NOT use this for simple single-file reads or single commands -
    use read/bash/ls/write/edit directly for those."""
    pass  # never actually runs - the router intercepts this before it's called


tools = [bash, read, write, edit, ls, explore]
tools_by_name = {t.name: t for t in tools}


def run_tool_call(call: dict) -> ToolMessage:
    """
    BUG FIX 2: runs ONE tool call safely and always returns a ToolMessage.

    Before, if the model called a tool wrongly (for example `edit` without
    `new_text`), Python raised an error and the WHOLE task crashed.
    Now the error is sent back to the model as a normal tool result,
    so it can read what went wrong and try again - like real agents do.
    """
    tool_fn = tools_by_name.get(call["name"])
    if tool_fn is None:
        result = (f"ERROR: unknown tool '{call['name']}'. "
                  f"Available tools: {', '.join(tools_by_name)}")
    else:
        try:
            result = tool_fn.invoke(call["args"])
        except Exception as e:
            result = (f"ERROR: the {call['name']} tool failed: "
                      f"{type(e).__name__}: {e}. "
                      "Check the arguments and try again.")
    # ToolMessage is the format the model API expects for tool results -
    # a plain dict looks similar but will error or get ignored.
    return ToolMessage(content=str(result), tool_call_id=call["id"])


# ============================================================
# 5. MODEL - the brain, with tools attached
# ============================================================

# Set MODEL_PROVIDER to "anthropic" or "openai".
# The model name can be changed WITHOUT editing code: put AGENT_MODEL=...
# in your .env file. This makes it easy to compare models on Terminal-Bench.
MODEL_PROVIDER = "openai"

if MODEL_PROVIDER == "anthropic":
    llm = ChatAnthropic(
        model=os.getenv("AGENT_MODEL", "claude-sonnet-5-5"),
        max_tokens=4096,  # raised from 1000 - writing whole files needs room
    )
else:
    llm = ChatOpenAI(model=os.getenv("AGENT_MODEL", "gpt-4o-mini"))

llm_with_tools = llm.bind_tools(tools)

# The subagent gets the REAL tools only - not "explore" itself,
# so it can't recursively spawn another subagent.
investigation_tools = [bash, read, ls]
investigation_tools_llm = llm.bind_tools(investigation_tools)


# ============================================================
# 6. NODES - each does ONE job, only updates state
# ============================================================

def think(state: AgentState) -> AgentState:
    """
    THE BRAIN. Reads the whole notebook (all messages so far) and decides
    what to do next.

    - If the model asks for a tool  -> done = False (routing sends it to act)
    - If the model just writes text -> done = True  (task is finished)

    The system prompt is added fresh on every call (not saved in state),
    so compaction can never summarize it away.

    If there is a deadline, the brain is also told how much time is left,
    and in the last minute it is told to stop working and finish.
    """
    prompt = SYSTEM_PROMPT
    left = seconds_left()
    if left is not None:
        prompt += f"\nTime left for this task: about {int(left)} seconds."
        if left < WRAP_UP_SECONDS:
            prompt += ("\nTIME IS ALMOST UP. Do not start anything new. "
                       "Make sure the main result is saved, then reply "
                       "with a short summary and no tool call.")

    messages = [SystemMessage(content=prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    state["messages"].append(response)

    if response.tool_calls:
        state["done"] = False
    else:
        state["done"] = True

    return state


def act(state: AgentState) -> AgentState:
    """
    THE HANDS. Runs every tool the brain asked for in its last message,
    and writes each result back into the notebook as a ToolMessage.

    Also adds +1 to step_count. Stops the agent (done = True) when:
    - the step limit (MAX_STEPS) is reached, or
    - the time deadline has passed.

    Tools run through run_tool_call(), so a bad tool call becomes an error
    message for the model instead of crashing the whole task.
    """
    last_message = state["messages"][-1]

    for call in last_message.tool_calls:
        state["messages"].append(run_tool_call(call))

    state["step_count"] += 1

    if state["step_count"] >= MAX_STEPS:
        state["done"] = True

    left = seconds_left()
    if left is not None and left <= 0:
        state["done"] = True

    return state


def explore_agent(state: AgentState) -> AgentState:
    """
    THE HELPER WORKER (sub-agent). Used for big search jobs.

    It runs its own mini think/act loop in a SEPARATE message list
    (sub_messages). All the messy grep/read/ls steps happen there and are
    thrown away. Only ONE short summary goes back into the main notebook.
    This keeps the main conversation small and clean.

    It only gets bash, read and ls - not explore itself - so it can't
    create another helper inside itself.

    If the brain asked for other tools in the SAME message (for example
    explore + bash), those are run here too. Every tool request must get
    an answer, or the model API rejects the conversation.
    """
    last_message = state["messages"][-1]
    explore_call = next(
        c for c in last_message.tool_calls if c["name"] == "explore"
    )
    question = explore_call["args"].get("question", "")

    # Answer the other tool requests from the same message first.
    for call in last_message.tool_calls:
        if call["name"] != "explore":
            state["messages"].append(run_tool_call(call))

    sub_messages = [
        SystemMessage(content=(
            "You are a research helper. Investigate the question using "
            "bash, read and ls. Look at real files and command output, then "
            "reply with a short, factual summary of what you found."
        )),
        HumanMessage(content=question),
    ]
    sub_steps = 0

    while sub_steps < MAX_SUB_STEPS:
        left = seconds_left()
        if left is not None and left < WRAP_UP_SECONDS:
            break  # not enough time left - return what we have

        response = investigation_tools_llm.invoke(sub_messages)
        sub_messages.append(response)

        if not response.tool_calls:
            break  # helper thinks it has the answer

        for call in response.tool_calls:
            sub_messages.append(run_tool_call(call))

        sub_steps += 1

    # The last message is the helper's answer only if it's an AI text reply.
    last = sub_messages[-1]
    if getattr(last, "type", "") == "ai" and last.content and not last.tool_calls:
        final_summary = last.content
    else:
        final_summary = ("The helper stopped before finishing. Last output:\n"
                         + str(last.content)[:2000])

    # Only this ONE clean summary re-enters the main conversation -
    # the sub_messages list itself is discarded here.
    state["exploration_result"] = final_summary
    state["messages"].append(
        ToolMessage(content=str(final_summary), tool_call_id=explore_call["id"])
    )
    state["step_count"] += 1

    if state["step_count"] >= MAX_STEPS:
        state["done"] = True

    left = seconds_left()
    if left is not None and left <= 0:
        state["done"] = True

    return state


def compact(state: AgentState) -> AgentState:
    """
    THE CLEANER. Runs when the notebook gets too long (over
    COMPACTION_THRESHOLD messages).

    Keeps: the original task + the last 3 messages.
    Replaces everything in between with ONE short summary.

    Why: long chats cost more money and make the model reason worse
    ("context rot"). This keeps the agent sharp on long tasks.

    BUG FIX 1: the cut point is moved back so the kept part never STARTS
    with a tool result. A tool result must always come right after the AI
    message that asked for it (question before answer). Before this fix,
    the question was sometimes summarized away while its answer was kept,
    and the model API rejected the whole conversation.
    """
    msgs = state["messages"]
    cut = len(msgs) - 3
    while cut > 1 and isinstance(msgs[cut], ToolMessage):
        cut -= 1
    if cut <= 1:
        return state  # nothing safe to summarize yet

    # msgs[0] is the original task - it is kept, not summarized.
    messages_to_summarize = msgs[1:cut]
    recent_messages = msgs[cut:]

    summary_prompt = [
        HumanMessage(content=(
            "Summarize what has happened so far in this task - "
            "what was tried, what was found, what's still pending. "
            "Keep it short, a few sentences:\n\n"
            # keep only the newest 60,000 characters, so huge command
            # outputs can't overflow the model's memory
            + str(messages_to_summarize)[-60000:]
        ))
    ]
    # Plain llm (no tools) - this call only needs to write a summary.
    summary_response = llm.invoke(summary_prompt)

    original_task = state["messages"][0]
    summary_message = HumanMessage(
        content=f"[Earlier steps summarized]: {summary_response.content}"
    )

    state["messages"] = [original_task, summary_message] + recent_messages
    return state


# ============================================================
# 7. ROUTING - decides where to go next (NOT inside the nodes)
# ============================================================

def route_after_think(state: AgentState) -> str:
    """After think: finished -> end, 'explore' -> helper, any other tool -> act."""
    if state["done"]:
        return "end"

    last_message = state["messages"][-1]
    requested_tool = last_message.tool_calls[0]["name"]

    if requested_tool == "explore":
        return "explore_agent"
    return "act"


def route_after_act(state: AgentState) -> str:
    """After act: step limit hit -> end, notebook too long -> compact, else -> think."""
    if state["done"]:
        return "end"
    if len(state["messages"]) > COMPACTION_THRESHOLD:
        return "compact"
    return "think"


def route_after_explore(state: AgentState) -> str:
    """After explore_agent: same rules as after act."""
    if state["done"]:
        return "end"
    if len(state["messages"]) > COMPACTION_THRESHOLD:
        return "compact"
    return "think"


def route_after_compact(state: AgentState) -> str:
    """After compact: always go back to think."""
    return "think"


# ============================================================
# 8. BUILD THE GRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("think", think)
graph.add_node("act", act)
graph.add_node("explore_agent", explore_agent)
graph.add_node("compact", compact)

graph.set_entry_point("think")

graph.add_conditional_edges("think", route_after_think, {
    "act": "act",
    "explore_agent": "explore_agent",
    "end": END,
})

graph.add_conditional_edges("act", route_after_act, {
    "compact": "compact",
    "think": "think",
    "end": END,
})

graph.add_conditional_edges("explore_agent", route_after_explore, {
    "compact": "compact",
    "think": "think",
    "end": END,
})

graph.add_conditional_edges("compact", route_after_compact, {
    "think": "think",
})

# ------------------------------------------------------------
# Checkpointing - saves state to a local SQLite file after
# every node, so a crashed run can resume instead of restarting.
# ------------------------------------------------------------
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

app = graph.compile(checkpointer=checkpointer)


# ============================================================
# 9. RUN IT
# ============================================================

def run_agent(task: str, thread_id: str = "default-task", resume: bool = False):
    """
    Runs the agent on one task from the command line.

    thread_id: a label for this task's save slot (checkpointing).
    resume:    True = continue from the last saved step of this thread_id.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        # Each act/think is one graph step; give room for MAX_STEPS loops.
        "recursion_limit": MAX_STEPS * 3 + 10,
    }

    if resume:
        # Don't pass a fresh state - LangGraph loads the last saved
        # checkpoint for this thread_id and continues from there.
        final_state = app.invoke(None, config=config)
    else:
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "step_count": 0,
            "done": False,
            "exploration_result": "",
        }
        final_state = app.invoke(initial_state, config=config)

    print("\n--- FINAL ANSWER ---")
    last_ai_message = final_state["messages"][-1]
    print(getattr(last_ai_message, "content", last_ai_message))


if __name__ == "__main__":
    run_agent("List the files in this folder", thread_id="task-001")
