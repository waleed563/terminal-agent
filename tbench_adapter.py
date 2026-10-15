"""
Terminal-Bench adapter.

This wraps your existing LangGraph agent (terminal_agent.py) so
Terminal-Bench's harness can run it. It does this by flipping on
ACTIVE_TMUX_SESSION before running your graph, so your tools send
commands through Terminal-Bench's real session live, instead of
running them locally.

Run with (one task at a time):
    tb run --dataset-name terminal-bench-core --dataset-version 0.1.1 \
        --agent-import-path tbench_adapter:MyTerminalAgent \
        --task-id hello-world --n-concurrent 1
"""

import time
import uuid
from pathlib import Path

# In terminal-bench 0.2.18, AgentResult lives in the same file as BaseAgent
from terminal_bench.agents.base_agent import BaseAgent, AgentResult
from terminal_bench.terminal.tmux_session import TmuxSession

import terminal_agent
from terminal_agent import app, HumanMessage

DEFAULT_TIME_LIMIT_SEC = 360   # Terminal-Bench's default per-task agent limit
SAFETY_MARGIN = 0.85           # stop at 85% of the limit, to finish cleanly


def get_task_time_limit(logging_dir: Path | None) -> float:
    """
    Finds this task's real time limit (max_agent_timeout_sec in task.yaml).

    Terminal-Bench saves logs in runs/<run>/<task_id>/<trial>/agent-logs,
    so the task name is 2 folders up from logging_dir. The task's settings
    file is in the downloaded dataset under ~/.cache/terminal-bench.
    If anything goes wrong, use the default 6-minute limit.
    """
    try:
        from terminal_bench.handlers.trial_handler import Task

        task_id = Path(logging_dir).parent.parent.name
        cache = Path.home() / ".cache" / "terminal-bench"
        for task_yaml in cache.rglob(f"{task_id}/task.yaml"):
            return float(Task.from_yaml(task_yaml).max_agent_timeout_sec)
    except Exception:
        pass
    return DEFAULT_TIME_LIMIT_SEC


class MyTerminalAgent(BaseAgent):

    @staticmethod
    def name() -> str:
        return "my-terminal-agent"

    def perform_task(
        self,
        instruction: str,   # MUST be named "instruction" - the harness passes it by this exact name
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        """
        Called by Terminal-Bench once per task.

        1. Turns on "test mode" (ACTIVE_TMUX_SESSION), so every tool works
           inside the task's container instead of on your machine.
        2. Sets a DEADLINE from the task's own time limit, so the agent
           finishes before Terminal-Bench stops it.
        3. Runs the normal LangGraph agent on the task instruction.
        4. Turns test mode off again, even if something crashed.

        NOTE: ACTIVE_TMUX_SESSION and DEADLINE are shared globals, so run
        Terminal-Bench with --n-concurrent 1. Two tasks at once would
        overwrite each other.
        """
        time_limit = get_task_time_limit(logging_dir)

        # Flip the switches: tools now work inside the task's container,
        # and the agent knows when it must stop.
        terminal_agent.ACTIVE_TMUX_SESSION = session
        terminal_agent.DEADLINE = time.time() + time_limit * SAFETY_MARGIN

        try:
            initial_state = {
                "messages": [HumanMessage(content=instruction)],
                "task": instruction,
                "step_count": 0,
                "done": False,
                "exploration_result": "",
            }
            # The graph is compiled with a checkpointer, so every run
            # MUST have a thread_id - give each benchmark task its own.
            config = {
                "configurable": {"thread_id": f"tbench-{uuid.uuid4()}"},
                # LangGraph stops after 25 graph steps by default. Each
                # think + act loop uses 2 steps, so give room for MAX_STEPS.
                "recursion_limit": terminal_agent.MAX_STEPS * 3 + 10,
            }
            app.invoke(initial_state, config=config)
        finally:
            # Always reset, even if the run crashes, so a later normal
            # run_agent() call doesn't use a session or deadline that no
            # longer exists.
            terminal_agent.ACTIVE_TMUX_SESSION = None
            terminal_agent.DEADLINE = None

        return AgentResult(total_input_tokens=0, total_output_tokens=0)
