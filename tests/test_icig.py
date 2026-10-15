"""ICIG array contracts and optional tests using the user's local checkpoint."""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("PIL")

from fusion_agent.models.icig import inference  # noqa: E402


class AverageTileModel(torch.nn.Module):
    """Isolate tile assembly from learned network behavior."""

    def forward(self, infrared: torch.Tensor, visible: torch.Tensor) -> list[torch.Tensor]:
        return [(infrared + visible) / 2]


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (255, 257), (385, 449)])
def test_tile_assembly_identity_and_odd_dimensions(shape: tuple[int, int]) -> None:
    source = np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)
    original = source.copy()
    fused, count = inference.fuse_image(AverageTileModel(), source, source, torch.device("cpu"))
    assert fused.shape == shape
    assert fused.dtype == np.uint8
    # The source truncates the overlap-weighted float result, allowing one level.
    assert np.abs(fused.astype(int) - source).max() <= 1
    assert count == len(inference.tile_starts(shape[0])) * len(inference.tile_starts(shape[1]))
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("value", [0, 64, 255])
def test_constant_tiles(value: int) -> None:
    source = np.full((259, 265), value, dtype=np.uint8)
    fused, _ = inference.fuse_image(AverageTileModel(), source, source, torch.device("cpu"))
    assert np.abs(fused.astype(int) - value).max() <= 1


@pytest.mark.parametrize(
    "visible, infrared",
    [
        (np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.uint8)),
        (np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 3), dtype=np.float32)),
        (np.zeros((2, 3, 4), dtype=np.uint8), np.zeros((2, 3), dtype=np.uint8)),
        (np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8)),
        (np.zeros((2, 4), dtype=np.uint8), np.zeros((2, 3), dtype=np.uint8)),
        (np.full((2, 3), np.nan), np.zeros((2, 3), dtype=np.uint8)),
        ([1, 2], np.zeros((2, 3), dtype=np.uint8)),
    ],
)
def test_invalid_inputs_rejected_before_weights(visible: np.ndarray, infrared: np.ndarray) -> None:
    with pytest.raises(ValueError):
        inference.fuse(visible, infrared, Path("missing.pth"))


def test_rgb_and_prepared_luminance() -> None:
    visible = np.full((7, 9, 3), [200, 100, 20], dtype=np.uint8)
    infrared = np.full((7, 9), 40, dtype=np.uint8)
    prepared = np.full((7, 9), 100, dtype=np.uint8)
    result = inference.fuse_with_model(
        visible, infrared, AverageTileModel(), visible_luminance=prepared
    )
    np.testing.assert_array_equal(result["gray"], 70)
    assert result["color"].shape == visible.shape
    assert result["color"].dtype == np.uint8
    assert result["metadata"]["visible_luminance"] == "provided"
    with pytest.raises(ValueError, match="dimensions"):
        inference.fuse_with_model(
            visible, infrared, AverageTileModel(), visible_luminance=prepared[:, :-1]
        )


def test_nonfinite_prediction_rejected() -> None:
    class InvalidModel(torch.nn.Module):
        def forward(self, infrared: torch.Tensor, visible: torch.Tensor) -> list[torch.Tensor]:
            return [infrared * float("nan")]

    source = np.zeros((3, 5), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="non-finite"):
        inference.fuse_image(InvalidModel(), source, source, torch.device("cpu"))


def test_checkpoint_is_required_and_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="weights do not exist"):
        inference.load_model(tmp_path / "missing.pth")
    invalid = tmp_path / "incomplete.pth"
    torch.save({"not_a_network_parameter": torch.ones(1)}, invalid)
    with pytest.raises(RuntimeError, match="state_dict"):
        inference.load_model(invalid)


@pytest.mark.parametrize("threads", [0, -1, 1.5, True])
def test_invalid_thread_count(threads: int) -> None:
    source = np.zeros((1, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="threads"):
        inference.fuse(source, source, Path("missing.pth"), threads=threads)


@pytest.fixture(scope="module")
def local_model() -> torch.nn.Module:
    weights = os.environ.get("FUSION_ICIG_WEIGHTS")
    if not weights:
        pytest.skip("Set FUSION_ICIG_WEIGHTS to opt into trained-model contract tests.")
    torch.set_num_threads(2)
    return inference.load_model(Path(weights))


@pytest.mark.parametrize("value", [0, 127, 255])
def test_trained_model_constant_and_identity_contract(
    local_model: torch.nn.Module, value: int
) -> None:
    source = np.full((7, 9), value, dtype=np.uint8)
    output = inference.fuse_with_model(source, source, local_model)["gray"]
    assert output.shape == source.shape
    assert output.dtype == np.uint8
    assert np.isfinite(output).all()
    # Learned fusion is not specified to preserve identity or constant intensity.
    np.testing.assert_array_equal(source, value)


def test_trained_model_repeatability_and_odd_size(local_model: torch.nn.Module) -> None:
    source = np.random.default_rng(11).integers(0, 256, (17, 19), dtype=np.uint8)
    first = inference.fuse_with_model(source, source, local_model)["gray"]
    second = inference.fuse_with_model(source, source, local_model)["gray"]
    np.testing.assert_array_equal(first, second)
    assert first.shape == source.shape
