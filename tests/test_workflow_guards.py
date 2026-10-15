import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fusion_agent.models.runtime import ModelOptions
from fusion_agent.schemas import FusionPlan, selected_methods
from fusion_agent.tools.fusion import wavelet_fusion
from fusion_agent.tools.image_io import save_image


def test_reject_mismatched_fusion():
    with pytest.raises(ValueError, match="equal"):
        wavelet_fusion(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))


def test_wavelet_signed_details():
    visible = np.array([[0, 100], [0, 100]], np.uint8)
    infrared = np.full((2, 2), 50, np.uint8)
    np.testing.assert_array_equal(wavelet_fusion(visible, infrared), visible)


def test_prcv_groups_and_cli_preserve_local_checkpoint_option(tmp_path, monkeypatch):
    from fusion_agent import cli

    assert selected_methods("both") == FusionPlan().methods == ["weighted", "wavelet"]
    assert selected_methods("research") == ["icig", "letters", "prcv"]
    assert selected_methods("all") == ["weighted", "wavelet", "icig", "letters", "prcv"]
    checkpoint = str(tmp_path / "private-model.pth")
    captured = {}

    def run_workflow(*args, **kwargs):
        captured["plan"] = args[4]
        captured.update(kwargs)
        return SimpleNamespace(files={"report": "mock-report.md"})

    monkeypatch.setattr(cli, "run_workflow", run_workflow)
    assert (
        cli.main(
            [
                "--visible",
                "v.png",
                "--infrared",
                "i.png",
                "--method",
                "prcv",
                "--prcv-weights",
                checkpoint,
            ]
        )
        == 0
    )
    assert captured["plan"].methods == ["prcv"]
    assert captured["model_options"].prcv_weights == checkpoint
    with pytest.raises(ValueError):
        FusionPlan(methods=["prcv"], prcv_weights=checkpoint)


def test_prcv_checkpoint_resolution_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FUSION_PRCV_WEIGHTS", raising=False)
    default = Path("weights/prcv/model.pth")
    default.parent.mkdir(parents=True)
    default.write_bytes(b"mock default; not loaded")
    assert ModelOptions().checkpoint("prcv") == default
    environment = tmp_path / "environment.pth"
    environment.write_bytes(b"mock environment; not loaded")
    monkeypatch.setenv("FUSION_PRCV_WEIGHTS", str(environment))
    assert ModelOptions().checkpoint("prcv") == environment
    explicit = tmp_path / "explicit.pth"
    explicit.write_bytes(b"mock explicit; not loaded")
    assert ModelOptions(prcv_weights=str(explicit)).checkpoint("prcv") == explicit
    with pytest.raises(ValueError, match="FUSION_PRCV_WEIGHTS"):
        ModelOptions(prcv_weights=str(tmp_path / "missing.pth")).checkpoint("prcv")


def test_audit_unpaired_and_corrupt(tmp_path):
    spec = importlib.util.spec_from_file_location("audit", Path("scripts/audit.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for modality in ["visible", "infrared"]:
        (tmp_path / modality).mkdir()
    (tmp_path / "visible" / "missing.png").write_bytes(b"bad")
    for modality in ["visible", "infrared"]:
        (tmp_path / modality / "bad.png").write_bytes(b"bad")
    result = module.audit(tmp_path)
    assert any("Missing counterpart" in x for x in result["issues"])
    assert any("decode" in x for x in result["issues"])


def test_agent_cannot_claim_execution(tmp_path, monkeypatch):
    agents = pytest.importorskip("agents")
    from fusion_agent.agents.coordinator import AgentConclusion, run_agent_workflow

    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setattr(
        agents.Runner,
        "run_sync",
        lambda *a, **kw: SimpleNamespace(
            final_output=AgentConclusion(summary="Invented success", limitations=[])
        ),
    )
    with pytest.raises(RuntimeError, match="did not execute"):
        run_agent_workflow("unused.png", "unused.png", tmp_path, "compare")


def test_agent_method_and_resize_guards(tmp_path):
    pytest.importorskip("agents")
    from agents.tool_context import ToolContext

    from fusion_agent.agents.coordinator import build_agent

    path = tmp_path / "input.png"
    save_image(path, np.arange(16, dtype=np.uint8).reshape(4, 4))
    agent, state = build_agent(path, path, tmp_path, "compare", FusionPlan(methods=["weighted"]))

    async def exercise():
        context = ToolContext(
            context=None, tool_name="test", tool_call_id="offline", tool_arguments="{}"
        )
        await agent.tools[0].on_invoke_tool(context, "{}")
        for plan, error in [
            (FusionPlan(methods=["wavelet"]), "methods exceed"),
            (FusionPlan(methods=["prcv"]), "methods exceed"),
            (FusionPlan(methods=["weighted"], resize=True), "Resizing requires"),
        ]:
            message = await agent.tools[1].on_invoke_tool(
                context, json.dumps({"plan": plan.model_dump()})
            )
            assert error in message
        assert state["report"] is None

    asyncio.run(exercise())


def test_agent_error_messages_do_not_expose_unexpected_details():
    from fusion_agent.agents.coordinator import PlanConstraintError, tool_failure_message

    assert "inspect_inputs" in tool_failure_message(
        None, PlanConstraintError("Call inspect_inputs before execute_fusion.")
    )
    for error in [ValueError("private-input-path"), RuntimeError("provider-private-payload")]:
        message = tool_failure_message(None, error)
        assert str(error) not in message
        assert "Local tool execution failed" in message
