import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fusion_agent.models.runtime import ModelOptions
from fusion_agent.schemas import FusionPlan
from fusion_agent.tools.image_io import save_image


@pytest.fixture
def inputs(tmp_path):
    paths = [tmp_path / "v.png", tmp_path / "i.png"]
    for n, path in enumerate(paths):
        save_image(path, np.arange(64, dtype=np.uint8).reshape(8, 8) + n)
    return paths


def test_no_key_fallback(inputs, tmp_path, monkeypatch):
    from fusion_agent.agents.coordinator import run_agent_workflow

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report = run_agent_workflow(*inputs, tmp_path, "compare")
    assert report.mode == "deterministic-fallback-no-key"


def test_sdk_tools_offline(inputs, tmp_path):
    pytest.importorskip("agents")
    from agents.tool_context import ToolContext

    from fusion_agent.agents.coordinator import build_agent

    agent, state = build_agent(*inputs, tmp_path, "compare", FusionPlan())
    assert agent.output_type.model_json_schema()["additionalProperties"] is False
    inspect_tool, execute_tool = agent.tools
    assert "plan" in execute_tool.params_json_schema["properties"]

    async def exercise():
        context = ToolContext(
            context=None,
            tool_name="execute_fusion",
            tool_call_id="offline-test",
            tool_arguments="{}",
        )
        early = await execute_tool.on_invoke_tool(context, json.dumps({"plan": {}}))
        assert "inspect" in early.lower()
        await inspect_tool.on_invoke_tool(context, "{}")
        response = await execute_tool.on_invoke_tool(
            context, json.dumps({"plan": FusionPlan().model_dump()})
        )
        assert "metrics" in json.loads(response)
        again = await execute_tool.on_invoke_tool(
            context, json.dumps({"plan": FusionPlan().model_dump()})
        )
        assert "Only one" in again

    asyncio.run(exercise())
    assert state["report"].mode == "agent"


def test_ui_callback(inputs, tmp_path):
    pytest.importorskip("gradio")
    from fusion_agent.ui.gradio_app import build_app, run_demo

    outputs = run_demo(*map(str, inputs), "Compare", "both", False, False, output=tmp_path)
    assert len(outputs[0]) == 2 and len(outputs[1]) == 2
    assert Path(outputs[-1]).is_file()
    components = build_app().get_config_file()["components"]
    methods = next(
        component for component in components if component["props"].get("label") == "Methods"
    )
    assert "prcv" in [value for _, value in methods["props"]["choices"]]


def test_prcv_agent_checkpoint_options_stay_local(inputs, tmp_path, monkeypatch):
    pytest.importorskip("agents")
    from agents.tool_context import ToolContext

    from fusion_agent.agents import coordinator

    options = ModelOptions(prcv_weights=str(tmp_path / "private-model.pth"))
    plan = FusionPlan(methods=["prcv"])
    captured = {}

    def local_workflow(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(metrics={}, recommendation="prcv", reason="Mock", limitations=[])

    monkeypatch.setattr(coordinator, "run_workflow", local_workflow)
    agent, state = coordinator.build_agent(*inputs, tmp_path, "compare", plan, options)
    inspect_tool, execute_tool = agent.tools
    assert "prcv_weights" not in json.dumps(execute_tool.params_json_schema)
    assert options.prcv_weights not in agent.instructions

    async def exercise():
        context = ToolContext(
            context=None,
            tool_name="execute_fusion",
            tool_call_id="offline-prcv",
            tool_arguments="{}",
        )
        await inspect_tool.on_invoke_tool(context, "{}")
        response = await execute_tool.on_invoke_tool(
            context, json.dumps({"plan": plan.model_dump()})
        )
        assert json.loads(response)["recommendation"] == "prcv"
        assert options.prcv_weights not in response

    asyncio.run(exercise())
    assert captured["model_options"] is options
    assert state["report"] is not None


def test_ui_missing_input():
    gr = pytest.importorskip("gradio")
    from fusion_agent.ui.gradio_app import run_demo

    with pytest.raises(gr.Error):
        run_demo(None, None, "task", "both", False, False)
