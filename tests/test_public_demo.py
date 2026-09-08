import hashlib
import json
from pathlib import Path

import pytest

from fusion_agent.tools.image_io import inspect_image_pair


def test_tno_provenance():
    root = Path("examples/tno")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["license"] == "CC BY 4.0"
    assert len(manifest["pairs"]) == 3
    for pair in manifest["pairs"]:
        paths = []
        for modality in ["visible", "infrared"]:
            record = pair[modality]
            path = root / record["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            paths.append(path)
        assert inspect_image_pair(*paths).same_size


def test_public_mode_rejects_agent(monkeypatch):
    gr = pytest.importorskip("gradio")
    from fusion_agent.ui.gradio_app import run_demo

    monkeypatch.setenv("FUSION_PUBLIC_DEMO", "1")
    with pytest.raises(gr.Error, match="disabled"):
        run_demo("examples/visible.png", "examples/infrared.png", "task", "both", False, True)


def test_clickable_examples_and_public_callback(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from fusion_agent.ui import gradio_app

    pairs = gradio_app.example_pairs()
    assert len(pairs) == 4
    monkeypatch.setenv("FUSION_PUBLIC_DEMO", "1")
    monkeypatch.setattr(gradio_app, "public_output_root", lambda: tmp_path)
    result = gradio_app.run_demo(*pairs[0], "Compare", "both", False, False)
    assert len(result[0]) == 2 and Path(result[-1]).is_file()
