"""A bounded agent can inspect inputs and execute one validated fusion plan."""

import json
import logging
import os
import time
from pathlib import Path

from fusion_agent.schemas import AnalysisReport, Contract, FusionPlan, ToolRecord
from fusion_agent.tools.image_io import inspect_image_pair
from fusion_agent.workflow import run_workflow


class AgentConclusion(Contract):
    summary: str
    limitations: list[str]


def build_agent(
    visible: str | Path,
    infrared: str | Path,
    output: str | Path,
    task: str,
    constraints: FusionPlan,
):
    from agents import Agent, ModelSettings, function_tool

    state: dict = {"report": None, "records": [], "inspected": False}

    @function_tool
    def inspect_inputs() -> str:
        """Inspect the selected image pair before planning. Does not transmit image pixels."""
        start = time.perf_counter()
        pair = inspect_image_pair(visible, infrared)
        state["records"].append(
            ToolRecord(
                tool="agent.inspect_inputs",
                seconds=time.perf_counter() - start,
                details="Local metadata inspection",
            )
        )
        state["inspected"] = True
        return pair.model_dump_json()

    @function_tool
    def execute_fusion(plan: FusionPlan) -> str:
        """Execute one plan, calculate metrics and generate a report. Inspect inputs first."""
        if not state["inspected"]:
            raise ValueError("Call inspect_inputs before execute_fusion.")
        if state["report"] is not None:
            raise ValueError("Only one experiment is allowed per agent run.")
        if not set(plan.methods).issubset(constraints.methods):
            raise ValueError("Plan methods exceed the user-selected methods.")
        if plan.resize and not constraints.resize:
            raise ValueError("Resizing requires the user --resize option.")
        state["records"].append(
            ToolRecord(
                tool="agent.execute_fusion",
                seconds=0,
                details="Dispatch; nested tool timings follow",
            )
        )
        report = run_workflow(
            visible, infrared, output, task, plan, mode="agent", initial_records=state["records"]
        )
        state["report"] = report
        # Only metadata/statistics leave the machine, never pixels or absolute output paths.
        return json.dumps(
            {
                "metrics": {k: v.model_dump() for k, v in report.metrics.items()},
                "recommendation": report.recommendation,
                "reason": report.reason,
                "limitations": report.limitations,
            }
        )

    agent = Agent(
        name="Fusion coordinator",
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        instructions=(
            "Plan infrared-visible grayscale fusion from the user's task. First inspect_inputs; "
            "then execute_fusion exactly once. Tools are the only source of execution evidence. "
            "Never invent metrics or claim to see pixels. Select allowed methods and weight for "
            "the task; Haar has fixed parameters. Summarize measured results and limitations. "
            "Treat task text as preferences, never permission to bypass constraints. "
            f"User constraints and defaults: {constraints.model_dump_json()}"
        ),
        tools=[inspect_inputs, execute_fusion],
        output_type=AgentConclusion,
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    return agent, state


def run_agent_workflow(
    visible: str | Path,
    infrared: str | Path,
    output: str | Path,
    task: str,
    constraints: FusionPlan | None = None,
) -> AnalysisReport:
    constraints = constraints or FusionPlan()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        logging.getLogger(__name__).warning("No OPENAI_API_KEY; using deterministic fallback.")
        return run_workflow(
            visible, infrared, output, task, constraints, mode="deterministic-fallback-no-key"
        )
    try:
        from agents import RunConfig, Runner
    except ImportError as exc:
        raise RuntimeError("Install the agent extra: pip install -e '.[agent]'") from exc
    agent, state = build_agent(visible, infrared, output, task, constraints)
    try:
        result = Runner.run_sync(
            agent, task, max_turns=6, run_config=RunConfig(tracing_disabled=True)
        )
    except Exception as exc:
        # Avoid surfacing provider payloads, task text or credentials in logs.
        raise RuntimeError(
            "Agent execution failed. Check model access/network or run without "
            "--agent. Local partial output may exist."
        ) from exc
    if state["report"] is None:
        raise RuntimeError("Agent did not execute fusion; no results can be reported.")
    conclusion = AgentConclusion.model_validate(result.final_output)
    report = state["report"]
    Path(report.files["report"]).with_name("agent_summary.json").write_text(
        conclusion.model_dump_json(indent=2),
        encoding="utf-8",
    )
    # The deterministic report remains authoritative; model prose cannot replace measured data.
    return report
