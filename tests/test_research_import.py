"""Archive import boundaries using synthetic bytes, without trained checkpoints."""

import hashlib
import importlib.util
import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

MEMBER = "PRCV_Fusion_submit/weights/model.pth"
NESTED = "JDFusion/PRCV_Fusion_submit.zip"
CHECKPOINT_LIMIT = 16 * 1024 * 1024
NESTED_LIMIT = 32 * 1024 * 1024
REVIEWED = b"synthetic checkpoint bytes; never loaded by PyTorch"


@pytest.fixture
def importer(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "import_research_weights.py"
    spec = importlib.util.spec_from_file_location("research_import_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "CHECKPOINTS",
        {"prcv": (MEMBER, "model.pth", hashlib.sha256(REVIEWED).hexdigest())},
    )
    return module


def archive_bytes(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(MEMBER, payload)
        archive.writestr("PRCV_Fusion_submit/infer.py", "raise RuntimeError('do not execute')")
        archive.writestr("../../outside.txt", "not an allowed extraction target")
    return buffer.getvalue()


def write_archive(tmp_path: Path, payload: bytes, *, nested: bool) -> Path:
    path = tmp_path / ("bundle.zip" if nested else "prcv.zip")
    data = archive_bytes(payload)
    if nested:
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(NESTED, data)
    else:
        path.write_bytes(data)
    return path


@pytest.mark.parametrize("nested", [False, True])
def test_prcv_direct_and_nested_import_exact_member_only(tmp_path, importer, nested):
    archive = write_archive(tmp_path, REVIEWED, nested=nested)
    destination = tmp_path / "weights"
    target = importer.import_checkpoint(archive, "prcv", destination)
    assert target == destination / "prcv" / "model.pth"
    assert target.read_bytes() == REVIEWED
    assert [path.relative_to(destination) for path in destination.rglob("*") if path.is_file()] == [
        Path("prcv/model.pth")
    ]
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("nested", [False, True])
def test_prcv_hash_mismatch_creates_no_destination(tmp_path, importer, nested):
    archive = write_archive(tmp_path, b"different synthetic checkpoint", nested=nested)
    destination = tmp_path / "weights"
    with pytest.raises(ValueError, match="does not match the reviewed research archive"):
        importer.import_checkpoint(archive, "prcv", destination)
    assert not destination.exists()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("existing", [REVIEWED, b"different existing local checkpoint"])
def test_prcv_existing_file_is_never_rewritten(tmp_path, importer, nested, existing):
    archive = write_archive(tmp_path, REVIEWED, nested=nested)
    destination = tmp_path / "weights"
    target = destination / "prcv" / "model.pth"
    target.parent.mkdir(parents=True)
    target.write_bytes(existing)
    timestamp = target.stat().st_mtime_ns
    if existing == REVIEWED:
        assert importer.import_checkpoint(archive, "prcv", destination) == target
    else:
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            importer.import_checkpoint(archive, "prcv", destination)
    assert target.read_bytes() == existing
    assert target.stat().st_mtime_ns == timestamp


@pytest.mark.parametrize("nested", [False, True])
def test_prcv_oversized_checkpoint_rejected_before_hash_check(tmp_path, importer, nested):
    archive = write_archive(tmp_path, b"x" * (CHECKPOINT_LIMIT + 1), nested=nested)
    destination = tmp_path / "weights"
    with pytest.raises(ValueError, match="Checkpoint exceeds the known archive size limit"):
        importer.import_checkpoint(archive, "prcv", destination)
    assert not destination.exists()


def test_prcv_oversized_nested_member_rejected_before_opening(tmp_path, importer):
    path = tmp_path / "large-bundle.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(NESTED, b"not a ZIP" * (NESTED_LIMIT // 9 + 1))
    destination = tmp_path / "weights"
    with pytest.raises(ValueError, match="Nested PRCV archive exceeds the reviewed size limit"):
        importer.import_checkpoint(path, "prcv", destination)
    assert not destination.exists()
