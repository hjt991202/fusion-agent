import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fusion_agent.models.runtime import ModelOptions
from fusion_agent.schemas import FusionPlan, selected_methods
from fusion_agent.tools.image_io import save_image
from fusion_agent.workflow import run_workflow


def test_method_groups_and_configuration():
    assert selected_methods("both") == ["weighted", "wavelet"]
    assert selected_methods("research") == ["icig", "letters", "prcv"]
    assert len(selected_methods("all")) == 5
    with pytest.raises(ValueError):
        selected_methods("arbitrary_module")
    with pytest.raises(ValueError):
        ModelOptions(threads=0)
    with pytest.raises(ValueError):
        FusionPlan(methods=["unknown"])


def test_missing_checkpoint_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="not bundled in Git"):
        ModelOptions(icig_weights=str(tmp_path / "missing.pth")).checkpoint("icig")


def test_research_dispatch_and_evidence(tmp_path, monkeypatch):
    from fusion_agent import workflow

    visible = np.zeros((5, 7, 3), np.uint8)
    visible[:] = (13, 31, 89)  # BGR encoded by the project's image IO.
    infrared = np.full((5, 7), 40, np.uint8)
    luminance = np.full((5, 7), 70, np.uint8)
    for name, array in [("v", visible), ("i", infrared), ("y", luminance)]:
        save_image(tmp_path / f"{name}.png", array)
    checkpoint = tmp_path / "test.pth"
    checkpoint.write_bytes(b"opaque mock checkpoint; never loaded")
    calls = []

    def fuse(v, i, **kwargs):
        np.testing.assert_array_equal(v[0, 0], [89, 31, 13])
        np.testing.assert_array_equal(i, infrared)
        np.testing.assert_array_equal(kwargs["visible_luminance"], luminance)
        calls.append(kwargs)
        return {"gray": luminance, "color": v, "metadata": {"device": "mock"}}

    monkeypatch.setattr(workflow, "load_adapter", lambda _: SimpleNamespace(fuse=fuse))
    actual_version = workflow.importlib.metadata.version
    monkeypatch.setattr(
        workflow.importlib.metadata,
        "version",
        lambda name: "mock" if name in {"torch", "scipy", "einops"} else actual_version(name),
    )
    report = run_workflow(
        tmp_path / "v.png",
        tmp_path / "i.png",
        tmp_path / "runs",
        plan=FusionPlan(methods=["icig"]),
        model_options=ModelOptions(icig_weights=str(checkpoint), visible_y=str(tmp_path / "y.png")),
    )
    assert len(calls) == 1
    assert set(report.metrics) == {"icig"}
    assert Path(report.files["icig_color"]).is_file()
    config = json.loads(Path(report.files["config"]).read_text())
    assert len(config["models"]["icig"]["checkpoint"]["sha256"]) == 64
    assert config["models"]["icig"]["auxiliary_inputs"]["visible_y"]["name"] == "y.png"
    assert "icig_weights" not in config["preprocessing"]["model_options"]
    assert any(record.tool == "icig_fusion" for record in report.records)


def test_agent_fallback_preserves_local_model_options(tmp_path, monkeypatch):
    from fusion_agent.agents import coordinator

    options = ModelOptions(material_mode="valid")
    captured = {}

    def workflow(*args, **kwargs):
        captured.update(kwargs)
        return "local_result"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(coordinator, "run_workflow", workflow)
    result = coordinator.run_agent_workflow(
        "v.png",
        "i.png",
        tmp_path,
        "task",
        FusionPlan(methods=["letters"]),
        model_options=options,
    )
    assert result == "local_result"
    assert captured["model_options"] is options
    assert captured["mode"] == "deterministic-fallback-no-key"


@pytest.mark.parametrize("color_ir", [False, True])
@pytest.mark.parametrize("resize", [False, True])
@pytest.mark.parametrize("method", ["letters", "prcv"])
def test_models_preserve_source_color_and_expand_gray_visible(
    tmp_path, monkeypatch, color_ir, resize, method
):
    import cv2

    from fusion_agent import workflow

    visible = np.full((5, 7), 70, np.uint8)
    ir_shape = (3, 4) if resize else (5, 7)
    infrared = np.full(ir_shape + ((3,) if color_ir else ()), 40, np.uint8)
    if color_ir:
        infrared[:] = (17, 41, 96)
    save_image(tmp_path / "v.png", visible)
    save_image(tmp_path / "i.png", infrared)
    checkpoint = tmp_path / "opaque.pt"
    checkpoint.write_bytes(b"mock")

    def fuse(v, i, **kwargs):
        assert v.shape == (5, 7, 3)
        np.testing.assert_array_equal(v[:, :, 0], visible)
        expected = cv2.cvtColor(infrared, cv2.COLOR_BGR2RGB) if color_ir else infrared
        if resize:
            expected = cv2.resize(expected, (7, 5), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(i, expected)
        if method == "letters":
            assert kwargs["material_mode"] == "valid"
        else:
            assert "material_mode" not in kwargs
        return {"gray": visible, "color": v, "metadata": {"device": "mock"}}

    monkeypatch.setattr(workflow, "load_adapter", lambda _: SimpleNamespace(fuse=fuse))
    actual_version = workflow.importlib.metadata.version
    monkeypatch.setattr(
        workflow.importlib.metadata,
        "version",
        lambda name: "mock" if name in {"torch", "scipy", "einops"} else actual_version(name),
    )
    report = run_workflow(
        tmp_path / "v.png",
        tmp_path / "i.png",
        tmp_path / "runs",
        plan=FusionPlan(methods=[method], resize=resize),
        model_options=ModelOptions(**{f"{method}_weights": str(checkpoint)}, material_mode="valid"),
    )
    assert set(report.metrics) == {method}
    config = json.loads(Path(report.files["config"]).read_text())
    assert f"{method}_weights" not in config["preprocessing"]["model_options"]
    assert len(config["models"][method]["checkpoint"]["sha256"]) == 64
