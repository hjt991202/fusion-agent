"""Contracts for letters inference using small, explicitly untrained fixtures."""

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
Image = pytest.importorskip("PIL.Image")

from fusion_agent.models.letters import boundary_engine, color_transfer, fuse  # noqa: E402
from fusion_agent.models.letters.jdfusion.architecture import IDRNet  # noqa: E402
from fusion_agent.models.letters.masked_context import masked_close, normalized_blur  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Small random weights test execution only, never trained-model quality."""
    path = tmp_path_factory.mktemp("letters") / "untrained.pt"
    with torch.random.fork_rng():
        torch.manual_seed(12)
        config = {"base_channels": 4, "flow_channels": 8}
        model = IDRNet(**config)
        torch.save({"model_config": config, "model_state": model.state_dict()}, path)
    return path


def pair(shape=(31, 27)):
    y, x = np.indices(shape)
    vis = np.stack((100 + x % 80, 65 + y % 50, 30 + (x + y) % 40), axis=2).astype(np.uint8)
    ir = np.uint8(160 - 60 * ((x % 9 < 3) & (y % 11 < 8)))
    return vis, ir


@pytest.mark.parametrize("shape", [(2, 2), (17, 19), (32, 35)])
def test_odd_and_small_shapes_preserved(checkpoint, shape):
    vis, ir = pair(shape)
    original = vis.copy(), ir.copy()
    result = fuse(vis, ir, checkpoint)
    assert result["color"].shape == (*shape, 3)
    assert result["gray"].shape == shape
    assert result["color"].dtype == result["gray"].dtype == np.uint8
    assert np.array_equal(result["gray"], np.asarray(Image.fromarray(result["color"]).convert("L")))
    assert result["metadata"]["input_color_order"] == "RGB"
    assert np.isfinite(result["metadata"]["flow_mean_pixels"])
    assert np.array_equal(vis, original[0])
    assert np.array_equal(ir, original[1])


@pytest.mark.parametrize("value", [0, 128, 255])
def test_identical_constants_execute_model_without_identity_shortcut(
    checkpoint, monkeypatch, value
):
    """Identity is not a learned-model contract; prove actual model execution."""
    calls = []
    infer = boundary_engine.infer

    def record(net, vis, ir):
        calls.append(net)
        return infer(net, vis, ir)

    monkeypatch.setattr(boundary_engine, "infer", record)
    vis = np.full((19, 21, 3), value, np.uint8)
    ir = np.full((19, 21), value, np.uint8)
    result = fuse(vis, ir, checkpoint, material_mode="valid")
    assert len(calls) == 1
    assert not calls[0].training
    assert result["metadata"]["material_source"] == "valid_support"
    assert np.isfinite(result["gray"]).all()


def test_repeatability_and_equivalent_ir_channels(checkpoint):
    vis, ir = pair()
    first = fuse(vis, ir, checkpoint)
    second = fuse(vis, np.repeat(ir[:, :, None], 3, axis=2), checkpoint)
    assert np.array_equal(first["color"], second["color"])
    first["metadata"]["refinement"]["anchor"]["ink_threshold"] = -1
    assert boundary_engine.CFG["anchor"]["ink_threshold"] == 0.42


@pytest.mark.parametrize(
    ("which", "value", "message"),
    [
        ("vis", np.zeros((3, 4), np.uint8), "RGB"),
        ("vis", np.zeros((3, 4, 4), np.uint8), "RGB"),
        ("vis", np.zeros((3, 4, 3), np.float32), "uint8"),
        ("vis", np.zeros((1, 4, 3), np.uint8), "at least 2"),
        ("vis", np.zeros((0, 4, 3), np.uint8), "at least 2"),
        ("vis", [[0, 0]], "uint8"),
        ("ir", np.full((3, 4), np.nan), "uint8"),
        ("ir", np.zeros((3, 4, 2), np.uint8), "grayscale or RGB"),
        ("ir", np.zeros((2, 4), np.uint8), "same pixel grid"),
    ],
)
def test_invalid_arrays_rejected_before_loading(which, value, message):
    vis, ir = pair((3, 4))
    with pytest.raises(ValueError, match=message):
        fuse(value if which == "vis" else vis, value if which == "ir" else ir, "missing.pt")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threads": 0}, "threads"),
        ({"threads": True}, "threads"),
        ({"threads": 33}, "threads"),
        ({"device": "invalid"}, "device"),
        ({"material_mode": "invalid"}, "material_mode"),
        ({"valid_mask": np.zeros((31, 27), bool)}, "valid support"),
        ({"valid_mask": np.ones((31, 27), float)}, "boolean or uint8"),
        ({"material_mask": np.ones((30, 27), bool)}, "height and width"),
        ({"material_mask": np.zeros((31, 27), bool)}, "Material support"),
    ],
)
def test_invalid_options_and_masks(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fuse(*pair(), "missing.pt", **kwargs)


def test_achromatic_requires_observed_material():
    with pytest.raises(ValueError, match="Cannot derive colored material"):
        fuse(np.full((5, 7, 3), 128, np.uint8), np.full((5, 7), 128, np.uint8), "missing.pt")


def test_masks_intersect_valid_support(checkpoint):
    vis, ir = pair()
    valid = np.ones(vis.shape[:2], np.uint8)
    valid[:5] = 0
    material = np.ones_like(valid)
    material[:, :4] = 0
    result = fuse(vis, ir, checkpoint, valid_mask=valid, material_mask=material)
    assert result["metadata"]["valid_pixels"] == int(valid.sum())
    assert result["metadata"]["material_pixels"] == int((valid & material).sum())
    assert result["metadata"]["material_source"] == "explicit_mask"
    with pytest.raises(ValueError, match="Material support"):
        fuse(vis, ir, checkpoint, valid_mask=valid, material_mask=1 - valid)


def test_missing_and_malformed_checkpoints(tmp_path, checkpoint):
    vis, ir = pair()
    with pytest.raises(FileNotFoundError, match="Missing letters checkpoint"):
        fuse(vis, ir, tmp_path / "missing.pt")
    malformed = tmp_path / "malformed.pt"
    torch.save({"model_state": {}}, malformed)
    with pytest.raises(ValueError, match="model_config and model_state"):
        fuse(vis, ir, malformed)
    data = torch.load(checkpoint, weights_only=True)
    data["model_state"].pop(next(iter(data["model_state"])))
    torch.save(data, malformed)
    with pytest.raises(RuntimeError, match="Missing key"):
        fuse(vis, ir, malformed)


def test_torch_thread_setting_restored_on_success_and_failure(checkpoint, tmp_path):
    previous = torch.get_num_threads()
    fuse(*pair(), checkpoint, threads=1)
    assert torch.get_num_threads() == previous
    with pytest.raises(FileNotFoundError):
        fuse(*pair(), tmp_path / "absent.pt", threads=1)
    assert torch.get_num_threads() == previous


@pytest.mark.parametrize("name", ["fused", "flow"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_model_outputs_rejected_before_quantization(name, value):
    class InvalidNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, vis, ir):
            result = {"fused": vis.clone(), "flow": torch.zeros_like(vis[:, :2])}
            result[name][0, 0, 0, 0] = value
            return result

    vis, ir = pair((3, 5))
    with pytest.raises(RuntimeError, match=f"non-finite {name} values"):
        boundary_engine.infer(InvalidNet(), vis, np.repeat(ir[:, :, None], 3, axis=2))


def test_nonfinite_checkpoint_rejected(checkpoint, tmp_path):
    data = torch.load(checkpoint, weights_only=True)
    key = next(iter(data["model_state"]))
    data["model_state"][key].flatten()[0] = float("nan")
    broken = tmp_path / "nonfinite.pt"
    torch.save(data, broken)
    with pytest.raises(ValueError, match="model_state contains non-finite"):
        fuse(*pair(), broken)


def test_color_transfer_preserves_integer_luminance_and_unmasked_pixels():
    rng = np.random.default_rng(4)
    vis = rng.integers(0, 256, (17, 19, 3), dtype=np.uint8)
    reference = rng.integers(0, 256, vis.shape, dtype=np.uint8)
    material = rng.random(vis.shape[:2]) > 0.4
    result = color_transfer.apply(vis, reference, material)
    assert np.array_equal(color_transfer.gray(result), color_transfer.gray(reference))
    assert np.array_equal(
        cv2.cvtColor(result, cv2.COLOR_RGB2GRAY), cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    )
    assert np.array_equal(result[~material], reference[~material])


def test_masked_morphology_constant_and_external_pixel_invariance():
    y, x = np.indices((96, 128))
    mask = y < 48 + np.rint(5 * np.sin(x / 12))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bright = np.where(mask, 65, 255).astype(np.float32)
    dark = np.where(mask, 65, 0).astype(np.float32)
    first = normalized_blur(masked_close(bright, mask, kernel), mask, 1)
    second = normalized_blur(masked_close(dark, mask, kernel), mask, 1)
    assert np.max(np.abs(first[mask] - 65)) < 0.0001
    assert np.array_equal(first[mask], second[mask])
