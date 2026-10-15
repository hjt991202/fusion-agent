"""PRCV contracts; synthetic weights exercise execution without quality claims."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from fusion_agent.models import prcv  # noqa: E402
from fusion_agent.models.prcv.memory import enable_window_chunking  # noqa: E402
from fusion_agent.models.prcv.model.architecture import (  # noqa: E402
    CrossMultiManifoldAttention,
    IDRNet,
    MultiManifoldAttention,
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """Use the exact architecture with random weights, not a substituted baseline."""
    path = tmp_path_factory.mktemp("prcv") / "untrained.pth"
    config = json.loads(Path(prcv.__file__).with_name("config.json").read_bytes())["model"]
    with torch.random.fork_rng():
        torch.manual_seed(21)
        torch.save(IDRNet(**config).state_dict(), path)
    return path


def pair(shape=(33, 35)) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(shape)
    visible = np.stack((90 + x % 120, 60 + y % 80, 20 + (x + y) % 60), axis=2).astype(np.uint8)
    infrared = np.uint8(180 - 80 * ((x % 9 < 4) & (y % 13 < 8)))
    return visible, infrared


@pytest.mark.parametrize("shape", [(33, 33), (35, 67), (65, 71)])
def test_network_shape_and_original_inputs_preserved(checkpoint: Path, shape) -> None:
    visible, infrared = pair(shape)
    original = visible.copy(), infrared.copy()
    result = prcv.fuse(visible, infrared, checkpoint)
    assert result["gray"].shape == shape
    assert result["color"].shape == (*shape, 3)
    assert result["gray"].dtype == result["color"].dtype == np.uint8
    np.testing.assert_array_equal(
        result["gray"], np.asarray(Image.fromarray(result["color"]).convert("L"))
    )
    np.testing.assert_array_equal(visible, original[0])
    np.testing.assert_array_equal(infrared, original[1])
    assert result["metadata"]["attention_mode"] == "full"
    assert result["metadata"]["flow_max_abs_component_pixels"] <= 3


@pytest.mark.parametrize("value", [0, 127, 255])
def test_identical_constant_inputs_execute_model(checkpoint: Path, monkeypatch, value: int) -> None:
    calls = []
    predict = prcv._predict

    def observed(*args):
        calls.append(args[0])
        return predict(*args)

    monkeypatch.setattr(prcv, "_predict", observed)
    visible = np.full((33, 35, 3), value, np.uint8)
    result = prcv.fuse(visible, visible[:, :, 0], checkpoint)
    assert len(calls) == 1 and not calls[0].training
    assert all(parameter.requires_grad for parameter in calls[0].parameters())
    assert np.isfinite(result["gray"]).all()
    # The learned/source refinement pipeline does not guarantee identity preservation.
    assert np.isfinite(result["metadata"]["flow_mean_pixels"])


def test_repeatability_rgb_ir_and_environment_mode(checkpoint: Path, monkeypatch) -> None:
    visible, infrared = pair()
    first = prcv.fuse(visible, infrared, checkpoint)
    monkeypatch.setenv("MMA_MODE", "euc")
    second = prcv.fuse(visible, np.repeat(infrared[:, :, None], 3, axis=2), checkpoint)
    np.testing.assert_array_equal(first["color"], second["color"])
    assert MultiManifoldAttention(8, 2)._mode() == "full"
    assert CrossMultiManifoldAttention(8, 2)._mode() == "full"


@pytest.mark.parametrize(
    "which, value, message",
    [
        ("visible", np.zeros((33, 35), np.uint8), "RGB"),
        ("visible", np.zeros((33, 35, 4), np.uint8), "RGB"),
        ("visible", np.zeros((33, 35, 3), np.float32), "uint8"),
        ("visible", np.zeros((32, 35, 3), np.uint8), "at least 33"),
        ("visible", np.zeros((0, 35, 3), np.uint8), "at least 33"),
        ("visible", [[1, 2]], "uint8"),
        ("infrared", np.zeros((33, 35, 2), np.uint8), "grayscale or RGB"),
        ("infrared", np.zeros((34, 35), np.uint8), "dimensions"),
        ("infrared", np.full((33, 35), np.nan), "uint8"),
    ],
)
def test_invalid_arrays_fail_before_loading(which, value, message) -> None:
    visible, infrared = pair()
    with pytest.raises(ValueError, match=message):
        prcv.fuse(
            value if which == "visible" else visible,
            value if which == "infrared" else infrared,
            "missing.pth",
        )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"threads": 0}, "threads"),
        ({"threads": True}, "threads"),
        ({"threads": 33}, "threads"),
        ({"window_batch": 0}, "window_batch"),
        ({"window_batch": True}, "window_batch"),
        ({"device": "invalid"}, "device"),
    ],
)
def test_invalid_options(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        prcv.fuse(*pair(), "missing.pth", **options)


def test_checkpoint_is_required_strict_and_finite(tmp_path: Path) -> None:
    path = tmp_path / "bad.pth"
    with pytest.raises(FileNotFoundError, match="Missing PRCV checkpoint"):
        prcv.load_model(path)
    torch.save({"unexpected": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="state_dict"):
        prcv.load_model(path)
    torch.save({"unexpected": torch.tensor(float("nan"))}, path)
    with pytest.raises(ValueError, match="non-finite"):
        prcv.load_model(path)
    torch.save({"unexpected": "not a tensor"}, path)
    with pytest.raises(ValueError, match="parameter names to tensors"):
        prcv.load_model(path)


@pytest.mark.parametrize("field", ["fused", "flow", "vfc_flow", "affine_theta"])
def test_nonfinite_outputs_rejected_before_quantization(field: str) -> None:
    class InvalidModel(torch.nn.Module):
        def forward(self, rgb, ir):
            result = {
                "fused": rgb,
                "flow": torch.zeros(1, 2, *ir.shape[-2:]),
                "vfc_flow": torch.zeros(1, 2, *ir.shape[-2:]),
                "affine_theta": torch.zeros(1, 2, 3),
            }
            result[field] = result[field] * float("nan")
            return result

    with pytest.raises(RuntimeError, match=f"non-finite {field}"):
        prcv._predict(InvalidModel(), *pair(), torch.device("cpu"))


def test_process_settings_restored_on_failure(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "not-loaded.pth"
    path.write_bytes(b"opaque mock data")
    before = (
        torch.get_num_threads(),
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )

    def fail(*args):
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(prcv, "_cached_model", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        prcv.fuse(*pair(), path, threads=1)
    assert before == (
        torch.get_num_threads(),
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )


@pytest.mark.parametrize("cross", [False, True])
def test_attention_chunking_preserves_math(cross: bool) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(2)
        model = (CrossMultiManifoldAttention if cross else MultiManifoldAttention)(8, 2).eval()
        inputs = [torch.randn(7, 9, 8)] * (2 if cross else 1)
        with torch.inference_mode():
            expected = model(*inputs)
            enable_window_chunking(model, 2)
            actual = model(*inputs)
            enable_window_chunking(model, 0)
            restored = model(*inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(restored, expected)
