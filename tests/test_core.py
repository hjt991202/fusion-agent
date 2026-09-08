import json
import subprocess
import sys

import numpy as np
import pytest

from fusion_agent.schemas import FusionPlan
from fusion_agent.tools.fusion import wavelet_fusion, weighted_average_fusion
from fusion_agent.tools.image_io import inspect_image, inspect_image_pair, read_image, save_image
from fusion_agent.tools.metrics import calculate_metrics
from fusion_agent.tools.registration import resize_or_align_pair
from fusion_agent.workflow import run_workflow


@pytest.fixture
def pair(tmp_path):
    v = np.arange(99, dtype=np.uint8).reshape(9, 11)
    i = np.flipud(v).copy()
    paths = [tmp_path / "visible.png", tmp_path / "infrared.png"]
    for path, image in zip(paths, [v, i], strict=True):
        save_image(path, image)
    return paths


def test_read_and_inspect(pair):
    assert read_image(pair[0]).shape == (9, 11)
    assert inspect_image_pair(*pair).same_size
    assert len(inspect_image(pair[0]).sha256) == 64


@pytest.mark.parametrize(
    "name,data",
    [("missing.png", None), ("bad.txt", b"x"), ("bad.png", b"not png"), ("empty.png", b"")],
)
def test_invalid_input(tmp_path, name, data):
    path = tmp_path / name
    if data is not None:
        path.write_bytes(data)
    with pytest.raises(ValueError):
        read_image(path)


def test_unsupported_depth_and_channels(tmp_path):
    for image in [np.zeros((3, 3), np.uint16), np.zeros((3, 3, 4), np.uint8)]:
        path = tmp_path / "unsupported.png"
        save_image(path, image)
        with pytest.raises(ValueError):
            read_image(path)


def test_constant_warning(tmp_path):
    path = tmp_path / "blank.png"
    save_image(path, np.zeros((3, 3), np.uint8))
    assert inspect_image(path).warnings


def test_resize():
    v, i = np.zeros((8, 8), np.uint8), np.ones((4, 5), np.uint8)
    with pytest.raises(ValueError, match="sizes differ"):
        resize_or_align_pair(v, i)
    assert resize_or_align_pair(v, i, resize=True)[1].shape == v.shape


def test_weighted():
    v, i = np.full((3, 3), 200, np.uint8), np.full((3, 3), 20, np.uint8)
    np.testing.assert_array_equal(weighted_average_fusion(v, i, 0.25), 65)
    np.testing.assert_array_equal(weighted_average_fusion(v, i, 1), v)
    np.testing.assert_array_equal(weighted_average_fusion(v, i, 0), i)
    with pytest.raises(ValueError):
        weighted_average_fusion(v, i, float("nan"))


@pytest.mark.parametrize("shape", [(8, 8), (9, 11), (2, 2)])
def test_wavelet_identity(shape):
    image = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    np.testing.assert_array_equal(wavelet_fusion(image, image), image)


def test_wavelet_constant():
    v, i = np.full((4, 4), 200, np.uint8), np.full((4, 4), 20, np.uint8)
    np.testing.assert_array_equal(wavelet_fusion(v, i), 110)


def test_metrics_analytic():
    zero = calculate_metrics(np.zeros((3, 3), np.uint8))
    assert all(v == 0 for v in zero.model_dump().values())
    ramp = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    result = calculate_metrics(ramp)
    assert result.entropy == 2
    assert result.standard_deviation == pytest.approx(np.sqrt(1.25))
    assert result.average_gradient == pytest.approx(np.sqrt(2.5))
    assert result.spatial_frequency == pytest.approx(np.sqrt(5))


def test_report_and_repeatability(pair, tmp_path):
    a = run_workflow(*pair, tmp_path / "out")
    b = run_workflow(*pair, tmp_path / "out")
    assert a.metrics == b.metrics
    assert a.files != b.files
    from pathlib import Path

    text = Path(a.files["report"]).read_text(encoding="utf-8")
    assert "## Metrics" in text and "wavelet" in text
    assert json.loads(Path(a.files["config"]).read_text())["seed"] == 0
    assert a.records[-1].tool == "generate_markdown_report"


def test_cli(pair, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fusion_agent.cli",
            "--visible",
            str(pair[0]),
            "--infrared",
            str(pair[1]),
            "--output",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "report.md" in result.stdout


def test_invalid_plan():
    with pytest.raises(ValueError):
        FusionPlan(methods=["invented"])
    with pytest.raises(ValueError):
        FusionPlan(visible_weight=2)
