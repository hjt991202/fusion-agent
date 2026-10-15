"""Verify the curated static release and its measured evidence without publishing."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def demo_module():
    pytest.importorskip("PIL")
    spec = importlib.util.spec_from_file_location("demo_jiandu", ROOT / "scripts/demo_jiandu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def previous_run_fixture(tmp_path):
    module = demo_module()
    examples, previous = tmp_path / "examples", tmp_path / "previous"
    identifier = "jiandu_0001"
    source = np.arange(63, dtype=np.uint8).reshape(7, 9) * 3
    pair = {"id": identifier, "source_id": 1}
    for key in module.MODALITIES:
        path = examples / key / f"{identifier}.png"
        path.parent.mkdir(parents=True)
        pixels = np.repeat(source[:, :, None], 3, axis=2) if key == "visible" else source
        module.Image.fromarray(pixels).save(path)
        pair[key] = {
            "file": f"{key}/{identifier}.png",
            "sha256": module.sha256(path),
            "width": 9,
            "height": 7,
        }
    (examples / "manifest.json").write_text(json.dumps({"pairs": [pair]}), encoding="utf-8")
    directory = previous / identifier / "run-fixture"
    directory.mkdir(parents=True)
    files, metrics = {}, {}
    for index, key in enumerate(module.PREVIOUS_METHODS):
        pixels = source + index
        path = directory / f"{key}.png"
        module.Image.fromarray(pixels).save(path)
        files[key] = str(path)
        metrics[key] = module.calculate_metrics(pixels).model_dump()
        if key in ["icig", "letters"]:
            color = directory / f"{key}_color.png"
            module.Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2)).save(color)
            files[f"{key}_color"] = str(color)
    timings = [
        {"tool": tool, "seconds": 0.1, "details": "Completed"}
        for tool in ["weighted_average_fusion", "wavelet_fusion", "icig_fusion", "letters_fusion"]
    ]
    plan = {"methods": module.PREVIOUS_METHODS, "resize": False}
    config = {
        "plan": plan,
        "input_hashes": [pair[key]["sha256"] for key in module.MODALITIES[:2]],
        "preprocessing": {"resize": "none"},
        "versions": {"fixture": "old"},
        "models": {
            "icig": {"auxiliary_inputs": {"visible_y": {"sha256": pair["visible_y"]["sha256"]}}},
            "letters": {},
        },
    }
    report = {
        "plan": plan,
        "metrics": metrics,
        "records": timings,
        "files": files,
        "inputs": {
            key: {"width": 9, "height": 7, "sha256": pair[key]["sha256"]}
            for key in module.MODALITIES[:2]
        },
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    record = {
        "id": identifier,
        "source_id": 1,
        "width": 9,
        "height": 7,
        "input_hashes": {key: pair[key]["sha256"] for key in module.MODALITIES},
        "metrics": metrics,
        "timings": timings,
        "seconds": 1.0,
        "models": config["models"],
        "versions": config["versions"],
        "output_hashes": {
            key: module.sha256(Path(files[key]))
            for key in ["icig", "icig_color", "letters", "letters_color"]
        },
        "clipping": {},
    }
    public = tmp_path / "previous-manifest.json"
    public.write_text(json.dumps({"pairs": [record]}), encoding="utf-8")
    (previous / "summary.json").write_text(json.dumps([record]), encoding="utf-8")
    return module, examples, previous, public, directory


def test_incremental_preflight_preserves_previous_config_and_checks_outputs(previous_run_fixture):
    module, examples, previous, public, directory = previous_run_fixture
    before = {path: path.read_bytes() for path in directory.iterdir()}
    pairs, _ = module.checked_inputs(examples)
    verified, paths = module.validate_previous_runs(pairs, previous, public)
    record = verified["jiandu_0001"]
    assert len(paths["jiandu_0001"]) == len(record["output_hashes"]) == 6
    assert record["runs"][0]["config"] == module.read_json(directory / "config.json")
    assert record["runs"][0]["versions"] == {"fixture": "old"}
    assert before == {path: path.read_bytes() for path in directory.iterdir()}


@pytest.mark.parametrize(
    "change", ["source_id", "source_hash", "report", "icig_png", "baseline_png", "ambiguous"]
)
def test_incremental_preflight_rejects_stale_or_changed_evidence(previous_run_fixture, change):
    module, examples, previous, public, directory = previous_run_fixture
    if change in ["source_id", "source_hash"]:
        records = module.read_json(public)["pairs"]
        if change == "source_id":
            records[0]["source_id"] = 2
        else:
            records[0]["input_hashes"]["visible_y"] = "0" * 64
        public.write_text(json.dumps({"pairs": records}), encoding="utf-8")
        (previous / "summary.json").write_text(json.dumps(records), encoding="utf-8")
    elif change == "report":
        report = module.read_json(directory / "report.json")
        report["metrics"]["icig"]["entropy"] += 1
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    elif change in ["icig_png", "baseline_png"]:
        path = directory / ("icig.png" if change == "icig_png" else "weighted.png")
        module.Image.fromarray(np.zeros((7, 9), dtype=np.uint8)).save(path)
    else:
        duplicate = directory.parent / "run-duplicate"
        duplicate.mkdir()
        shutil.copy2(directory / "report.json", duplicate / "report.json")
    pairs, _ = module.checked_inputs(examples)
    with pytest.raises(ValueError):
        module.validate_previous_runs(pairs, previous, public)


def test_incremental_main_executes_only_prcv_and_keeps_separate_run_evidence(
    previous_run_fixture, tmp_path, monkeypatch
):
    module, examples, previous, public, directory = previous_run_fixture
    old_bytes = {path: path.read_bytes() for path in directory.iterdir()}
    calls = []

    def fake_workflow(visible, infrared, output, *, plan, **kwargs):
        calls.append(plan.methods)
        run = output / "run-new-fixture"
        run.mkdir(parents=True)
        with module.Image.open(visible) as image:
            color = np.asarray(image)
        gray = np.asarray(module.Image.fromarray(color).convert("L"))
        module.Image.fromarray(color).save(run / "prcv_color.png")
        module.Image.fromarray(gray).save(run / "prcv.png")
        metric = module.calculate_metrics(gray)
        record = {"tool": "prcv_fusion", "seconds": 0.2, "details": "Fixture, not model inference"}
        config = {
            "plan": {"methods": plan.methods},
            "versions": {"fixture": "new"},
            "models": {"prcv": {"fixture": True}},
        }
        files = {
            "prcv": str(run / "prcv.png"),
            "prcv_color": str(run / "prcv_color.png"),
            "config": str(run / "config.json"),
            "json": str(run / "report.json"),
        }
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "report.json").write_text(
            json.dumps({"plan": config["plan"], "records": [record]}), encoding="utf-8"
        )
        return SimpleNamespace(
            files=files,
            metrics={"prcv": metric},
            records=[SimpleNamespace(tool="prcv_fusion", model_dump=lambda: record)],
        )

    monkeypatch.setattr(module, "run_workflow", fake_workflow)
    monkeypatch.setattr(module, "FusionPlan", lambda **kwargs: SimpleNamespace(**kwargs))
    output, assets = tmp_path / "new-output", tmp_path / "new-assets"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_jiandu.py",
            "--examples",
            str(examples),
            "--output",
            str(output),
            "--assets",
            str(assets),
            "--incremental-prcv",
            "--previous-output",
            str(previous),
            "--previous-manifest",
            str(public),
        ],
    )
    module.main()
    record = module.read_json(assets / "manifest.json")["pairs"][0]
    assert calls == [["prcv"]]
    assert "seconds" not in record and "versions" not in record
    assert record["timing_scope"] == "separate_runs"
    assert [run["versions"] for run in record["runs"]] == [{"fixture": "old"}, {"fixture": "new"}]
    assert record["runs"][0]["seconds"] == 1.0
    assert set(record["metrics"]) == set(module.ALL_METHODS)
    assert len(record["output_hashes"]) == 8
    with module.Image.open(assets / record["preview"]) as plate:
        assert plate.size == (1250, 740)
    assert old_bytes == {path: path.read_bytes() for path in directory.iterdir()}


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
