"""Keep the small, user-authorized demonstration subset traceable and unchanged."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def preparation_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_jiandu_examples", ROOT / "scripts" / "prepare_jiandu_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jiandu_manifest_integrity():
    root = ROOT / "examples" / "jiandu"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pair_count"] == len(manifest["pairs"]) == 15
    assert "No separate dataset license" in manifest["license"]
    assert "not a benchmark" in manifest["usage"]
    ids = [pair["id"] for pair in manifest["pairs"]]
    assert len(set(ids)) == len(ids)
    actual_bytes = 0
    expected_files = set()
    for pair in manifest["pairs"]:
        dimensions = set()
        for modality, channels in [("visible", 3), ("infrared", 1), ("visible_y", 1)]:
            record = pair[modality]
            relative = Path(record["file"])
            assert relative.parts == (modality, pair["id"] + ".png")
            expected_files.add(relative.as_posix())
            payload = (root / relative).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            assert digest == record["sha256"] == record["source_sha256"]
            assert len(payload) == record["bytes"]
            actual_bytes += len(payload)
            pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            assert pixels.dtype == np.uint8
            assert pixels.shape[:2] == (record["height"], record["width"])
            assert (1 if pixels.ndim == 2 else pixels.shape[2]) == channels == record["channels"]
            if channels == 3:
                pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == record["pixel_sha256"]
            dimensions.add(pixels.shape[:2])
        assert len(dimensions) == 1
    assert actual_bytes == manifest["total_image_bytes"]
    assert {path.relative_to(root).as_posix() for path in root.rglob("*.png")} == expected_files


def test_source_audit_reports_missing_and_invalid_images_without_modifying_sources(tmp_path):
    module = preparation_module()
    source = tmp_path / "test"
    image = np.arange(63, dtype=np.uint8).reshape(7, 9)
    for folder, prefix, channels in module.MODALITIES.values():
        directory = source / folder
        directory.mkdir(parents=True)
        array = np.repeat(image[:, :, None], channels, axis=2) if channels == 3 else image
        success, payload = cv2.imencode(".png", array)
        assert success
        (directory / f"{prefix}1.png").write_bytes(payload.tobytes())
    (source / "IR" / "IR2.png").write_bytes(b"not an image")
    before = {path: path.read_bytes() for path in source.rglob("*.png")}
    audit = module.audit_sources(source)
    assert audit["source_ids_checked"] == 2
    assert audit["eligible_pairs"] == 1
    assert any("could not be decoded" in issue for issue in audit["issues"])
    assert any("Missing counterpart" in issue for issue in audit["issues"])
    assert before == {path: path.read_bytes() for path in source.rglob("*.png")}


def test_preparation_rejects_output_inside_source(tmp_path):
    module = preparation_module()
    source = tmp_path / "test"
    with pytest.raises(ValueError, match="separate from the source dataset"):
        module.prepare(source, source / "selected", tmp_path / "audit", 15)
