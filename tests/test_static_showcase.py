"""Verify the curated static release and its measured evidence without publishing."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB_ASSETS = ["index.html", "app.mjs", "fusion.mjs", "showcase.mjs", "style.css"]


def staging_module():
    spec = importlib.util.spec_from_file_location(
        "deploy_static_space", ROOT / "scripts" / "deploy_static_space.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_fixture(tmp_path):
    root = tmp_path / "source"
    for relative in ["examples/jiandu/manifest.json", "docs/assets/jiandu/manifest.json"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return root


def test_actual_stage_contains_only_allowed_files_and_original_bytes(tmp_path):
    stage = tmp_path / "stage"
    staging_module().prepare_static(ROOT, stage)
    inputs = json.loads((ROOT / "examples/jiandu/manifest.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "docs/assets/jiandu/manifest.json").read_text(encoding="utf-8"))
    sources = {pair["id"]: pair for pair in inputs["pairs"]}
    expected = set(WEB_ASSETS) | {
        "LICENSE",
        "README.md",
        "samples/visible.png",
        "samples/infrared.png",
    }
    expected |= {
        "samples/jiandu/README.md",
        "samples/jiandu/manifest.json",
        "showcase/manifest.json",
    }
    for path in (ROOT / "examples/tno").rglob("*"):
        if path.is_file():
            relative = Path("samples/tno") / path.relative_to(ROOT / "examples/tno")
            expected.add(relative.as_posix())
            assert (stage / relative).read_bytes() == path.read_bytes()
    for pair in inputs["pairs"]:
        for modality in ["visible", "infrared", "visible_y"]:
            record = pair[modality]
            relative = Path("samples/jiandu") / record["file"]
            expected.add(relative.as_posix())
            assert hashlib.sha256((stage / relative).read_bytes()).hexdigest() == record["sha256"]
    for pair in results["pairs"]:
        source = sources[pair["id"]]
        assert pair["input_hashes"] == {
            key: source[key]["sha256"] for key in ["visible", "infrared", "visible_y"]
        }
        assert pair["source_id"] == source["source_id"]
        assert (pair["width"], pair["height"]) == (
            source["visible"]["width"],
            source["visible"]["height"],
        )
        relative = Path("showcase") / pair["preview"]
        expected.add(relative.as_posix())
        assert (stage / relative).read_bytes() == (
            ROOT / "docs/assets/jiandu" / pair["preview"]
        ).read_bytes()
    actual = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
    assert actual == expected
    assert len(list((stage / "samples/jiandu").rglob("*.png"))) == 45
    assert len(list((stage / "showcase").glob("*.png"))) == 15
    assert (stage / "samples/jiandu/README.md").read_bytes() == (
        ROOT / "examples/jiandu/README.md"
    ).read_bytes()


@pytest.mark.parametrize("change", ["source_hash", "source_id", "dimensions", "selected_ids"])
def test_stage_rejects_measurements_from_different_sources(tmp_path, change):
    root = manifest_fixture(tmp_path)
    path = root / "docs/assets/jiandu/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pair = manifest["pairs"][0]
    if change == "source_hash":
        pair["input_hashes"]["visible_y"] = "0" * 64
    elif change == "source_id":
        pair["source_id"] += 1
    elif change == "dimensions":
        pair["width"] += 1
    else:
        pair["id"] = "jiandu_9999"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="match"):
        staging_module().prepare_static(root, stage)
    assert not stage.exists()


@pytest.mark.parametrize("target", ["source", "preview"])
def test_stage_rejects_traversal_before_copying(tmp_path, target):
    root = manifest_fixture(tmp_path)
    relative = (
        "examples/jiandu/manifest.json"
        if target == "source"
        else "docs/assets/jiandu/manifest.json"
    )
    path = root / relative
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if target == "source":
        manifest["pairs"][0]["visible"]["file"] = "../../outside.png"
    else:
        manifest["pairs"][0]["preview"] = "../../outside.png"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="Unexpected .* path"):
        staging_module().prepare_static(root, stage)
    assert not stage.exists()


def test_stage_rejects_changed_image_bytes_before_copying(tmp_path):
    root = manifest_fixture(tmp_path)
    manifest = json.loads((root / "examples/jiandu/manifest.json").read_text(encoding="utf-8"))
    image = root / "examples/jiandu" / manifest["pairs"][0]["visible"]["file"]
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not the reviewed image bytes")
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="Curated input hash mismatch"):
        staging_module().prepare_static(root, stage)
    assert not stage.exists()


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node needed for browser showcase validation"
)
def test_browser_accepts_actual_evidence_and_rejects_incomplete_or_unsafe_records():
    javascript = """
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateShowcase} from './web/showcase.mjs';
const manifest=JSON.parse(readFileSync('docs/assets/jiandu/manifest.json','utf8'));
assert.equal(validateShowcase(manifest).length,15);
for(const mutate of [
  m=>m.pairs.pop(),
  m=>m.pairs[0].preview='../outside.png',
  m=>m.pairs[1].id=m.pairs[0].id,
  m=>m.pairs[0].metrics.icig.entropy=NaN,
  m=>m.pairs[0].timings.pop(),
]){
  const invalid=structuredClone(manifest);mutate(invalid);
  assert.throws(()=>validateShowcase(invalid));
}
"""
    subprocess.run(
        ["node", "--input-type=module", "-e", javascript],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
