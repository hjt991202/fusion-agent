"""Run the curated source-resolution examples and build measured demonstration previews."""

import argparse
import copy
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from fusion_agent.models.runtime import ModelOptions
from fusion_agent.schemas import FusionPlan
from fusion_agent.tools.metrics import calculate_metrics
from fusion_agent.workflow import run_workflow

MODALITIES = ["visible", "infrared", "visible_y"]
PREVIOUS_METHODS = ["weighted", "wavelet", "icig", "letters"]
RESEARCH_METHODS = ["icig", "letters", "prcv"]
ALL_METHODS = [*PREVIOUS_METHODS, "prcv"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(records: list[dict]) -> dict[str, dict]:
    result = {}
    for record in records:
        identifier = record.get("id", "")
        if not re.fullmatch(r"jiandu_\d{4,}", identifier) or identifier in result:
            raise ValueError("Unsafe or duplicate example identifier")
        result[identifier] = record
    return result


def checked_inputs(examples: Path) -> tuple[list[dict], dict[str, dict[str, Path]]]:
    pairs = read_json(examples / "manifest.json")["pairs"]
    indexed(pairs)
    paths = {}
    for pair in pairs:
        identifier = pair["id"]
        paths[identifier] = {}
        for modality in MODALITIES:
            record = pair[modality]
            relative = Path(record["file"])
            if relative.parts != (modality, f"{identifier}.png"):
                raise ValueError("Example path must remain inside its selected modality")
            path = (examples / relative).resolve()
            if not path.is_relative_to(examples.resolve()) or sha256(path) != record["sha256"]:
                raise ValueError("Example hash/path does not match the manifest")
            with Image.open(path) as image:
                if image.size != (pair["visible"]["width"], pair["visible"]["height"]):
                    raise ValueError("Example dimensions do not match the manifest")
            paths[identifier][modality] = path
    return pairs, paths


def output_evidence(paths: dict[str, Path], width: int, height: int) -> tuple[dict, dict]:
    hashes, clipping = {}, {}
    for key, path in paths.items():
        hashes[key] = sha256(path)
        with Image.open(path) as image:
            array = np.asarray(image)
            if array.shape[:2] != (height, width) or array.dtype != np.uint8:
                raise ValueError("Output dimensions or dtype changed")
            clipping[key] = {
                "fraction_at_zero": float(np.mean(array == 0)),
                "fraction_at_255": float(np.mean(array == 255)),
            }
    return hashes, clipping


def run_evidence(label: str, report_path: Path, config_path: Path, seconds: float) -> dict:
    report, config = read_json(report_path), read_json(config_path)
    return {
        "label": label,
        "run_id": report_path.parent.name,
        "methods": report["plan"]["methods"],
        "seconds": seconds,
        "timings": [item for item in report["records"] if item["tool"].endswith("fusion")],
        "versions": config["versions"],
        "config": config,
        "report_sha256": sha256(report_path),
        "config_sha256": sha256(config_path),
    }


def validate_previous_runs(
    pairs: list[dict], previous_output: Path, previous_manifest: Path
) -> tuple[dict[str, dict], dict[str, dict[str, Path]]]:
    """Bind prior records to unchanged inputs, local reports, configs and saved output pixels."""
    records = indexed(read_json(previous_manifest)["pairs"])
    summaries = indexed(read_json(previous_output / "summary.json"))
    if records.keys() != indexed(pairs).keys() or summaries != records:
        raise ValueError("Previous manifest and run summaries do not match selected source IDs")
    verified, output_paths = {}, {}
    for pair in pairs:
        identifier = pair["id"]
        old = records[identifier]
        if (
            old["source_id"] != pair["source_id"]
            or (old["width"], old["height"])
            != (pair["visible"]["width"], pair["visible"]["height"])
            or old["input_hashes"] != {key: pair[key]["sha256"] for key in MODALITIES}
            or set(old["metrics"]) != set(PREVIOUS_METHODS)
        ):
            raise ValueError("Previous evidence does not match selected inputs or four methods")
        candidates = sorted((previous_output / identifier).glob("run-*/report.json"))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one unambiguous previous run per pair")
        report_path = candidates[0]
        if not report_path.resolve().is_relative_to(previous_output.resolve()):
            raise ValueError("Previous run path escaped its output directory")
        directory = report_path.parent
        config_path = directory / "config.json"
        report, config = read_json(report_path), read_json(config_path)
        timings = [item for item in report["records"] if item["tool"].endswith("fusion")]
        if (
            report["metrics"] != old["metrics"]
            or timings != old["timings"]
            or config["models"] != old["models"]
            or config["versions"] != old["versions"]
            or config["plan"] != report["plan"]
            or set(report["plan"]["methods"]) != set(PREVIOUS_METHODS)
            or config["preprocessing"]["resize"] != "none"
            or report["plan"]["resize"]
            or config["input_hashes"] != [pair[key]["sha256"] for key in MODALITIES[:2]]
            or config["models"]["icig"]["auxiliary_inputs"]["visible_y"]["sha256"]
            != pair["visible_y"]["sha256"]
        ):
            raise ValueError("Previous report/config does not match measured manifest evidence")
        for key in MODALITIES[:2]:
            info = report["inputs"][key]
            if info["sha256"] != pair[key]["sha256"] or (info["width"], info["height"]) != (
                old["width"],
                old["height"],
            ):
                raise ValueError("Previous report input identity changed")
        keys = [*PREVIOUS_METHODS, "icig_color", "letters_color"]
        paths = {}
        for key in keys:
            if Path(report["files"][key]).name != f"{key}.png":
                raise ValueError("Unexpected previous output filename")
            path = directory / f"{key}.png"
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Previous output path escaped its run directory")
            paths[key] = path
        hashes, clipping = output_evidence(paths, old["width"], old["height"])
        required_hashes = {"icig", "icig_color", "letters", "letters_color"}
        if not required_hashes.issubset(old["output_hashes"]) or any(
            hashes.get(key) != digest for key, digest in old["output_hashes"].items()
        ):
            raise ValueError("Previous output PNG hash mismatch")
        for key in PREVIOUS_METHODS:
            with Image.open(paths[key]) as image:
                actual = calculate_metrics(np.asarray(image)).model_dump()
            if any(
                not np.isclose(value, old["metrics"][key][metric], rtol=1e-12, atol=1e-12)
                for metric, value in actual.items()
            ):
                raise ValueError("Previous saved output metrics do not match measured evidence")
        record = copy.deepcopy(old)
        record["output_hashes"] = hashes
        record["clipping"] = clipping
        record["runs"] = [
            run_evidence("previous_four_methods", report_path, config_path, old["seconds"])
        ]
        record["previous_output_verification"] = (
            "ICIG/letters gray/color PNG hashes match the previous manifest. All four saved "
            "gray metrics were recomputed. Baseline PNG hashes are first recorded at this merge."
        )
        verified[identifier], output_paths[identifier] = record, paths
    return verified, output_paths


def preview(paths: list[Path], labels: list[str], target: Path) -> None:
    """Show entire images plus identical center crops; only the display is rescaled."""
    plate = Image.new("RGB", (250 * len(paths), 740), "#edf1f7")
    draw = ImageDraw.Draw(plate)
    for index, (path, label) in enumerate(zip(paths, labels, strict=True)):
        with Image.open(path) as source:
            image = source.convert("RGB")
        x = index * 250
        draw.text((x + 14, 12), label, fill="#183153")
        thumb = ImageOps.contain(image, (230, 410), Image.Resampling.LANCZOS)
        plate.paste(thumb, (x + (250 - thumb.width) // 2, 40 + (410 - thumb.height) // 2))
        side = min(256, *image.size)
        left, top = (image.width - side) // 2, (image.height - side) // 2
        crop = image.crop((left, top, left + side, top + side))
        crop = crop.resize((220, 220), Image.Resampling.NEAREST)
        plate.paste(crop, (x + 15, 490))
        draw.text((x + 14, 466), f"Center crop: {side} x {side}px", fill="#183153")
    draw.text(
        (14, 722),
        "Top: display-scaled full image. Bottom: same source coordinates.",
        fill="#183153",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    plate.save(target)


def merge_prcv_record(previous: dict, current: dict) -> dict:
    """Retain separate execution/configuration evidence rather than invent a combined duration."""
    if any(
        previous[key] != current[key]
        for key in ["id", "source_id", "width", "height", "input_hashes"]
    ):
        raise ValueError("Incremental PRCV run does not match the previous input identity")
    if set(current["metrics"]) != {"prcv"} or set(current["models"]) != {"prcv"}:
        raise ValueError("Incremental run must execute PRCV only")
    merged = copy.deepcopy(previous)
    for key in ["metrics", "models", "output_hashes", "clipping"]:
        merged[key].update(current[key])
    merged["timings"] += current["timings"]
    merged["runs"] += current["runs"]
    merged["timing_scope"] = "separate_runs"
    merged["timing_note"] = (
        "Previous four methods and PRCV ran in separate invocations. Per-run seconds, "
        "versions and full configurations are retained in runs; no combined wall time is claimed."
    )
    merged.pop("seconds", None)
    merged.pop("versions", None)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=Path("examples/jiandu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--assets", type=Path, help="Publish previews locally only after every run succeeds"
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--incremental-prcv", action="store_true", help="Verify and reuse earlier four-method runs"
    )
    parser.add_argument("--previous-output", type=Path)
    parser.add_argument("--previous-manifest", type=Path)
    args = parser.parse_args()
    if args.incremental_prcv != bool(args.previous_output and args.previous_manifest):
        parser.error("--incremental-prcv requires both --previous-output and --previous-manifest")
    if not args.incremental_prcv and (args.previous_output or args.previous_manifest):
        parser.error("Previous-run options require --incremental-prcv")
    pairs, input_paths = checked_inputs(args.examples)
    prior, prior_paths = {}, {}
    if args.incremental_prcv:
        prior, prior_paths = validate_previous_runs(
            pairs, args.previous_output, args.previous_manifest
        )
    args.output.mkdir(parents=True, exist_ok=False)
    methods = ["prcv"] if args.incremental_prcv else ALL_METHODS
    records, all_output_paths = [], {}
    for pair in pairs:
        identifier, paths = pair["id"], input_paths[pair["id"]]
        started = time.perf_counter()
        report = run_workflow(
            paths["visible"],
            paths["infrared"],
            args.output / identifier,
            task=f"Jiandu demonstration {identifier}",
            plan=FusionPlan(
                methods=methods, rationale="Run the requested measured demonstration methods."
            ),
            model_options=ModelOptions(
                visible_y=str(paths["visible_y"]), device=args.device, threads=args.threads
            ),
        )
        elapsed = time.perf_counter() - started
        config = read_json(Path(report.files["config"]))
        keys = [*methods, *(f"{method}_color" for method in methods if method in RESEARCH_METHODS)]
        paths_out = {key: Path(report.files[key]) for key in keys}
        hashes, clipping = output_evidence(
            paths_out, pair["visible"]["width"], pair["visible"]["height"]
        )
        record = {
            "id": identifier,
            "source_id": pair["source_id"],
            "width": pair["visible"]["width"],
            "height": pair["visible"]["height"],
            "input_hashes": {key: pair[key]["sha256"] for key in MODALITIES},
            "metrics": {key: value.model_dump() for key, value in report.metrics.items()},
            "seconds": elapsed,
            "timings": [
                item.model_dump() for item in report.records if item.tool.endswith("fusion")
            ],
            "models": config["models"],
            "versions": config["versions"],
            "output_hashes": hashes,
            "clipping": clipping,
            "timing_scope": "single_run",
            "runs": [
                run_evidence(
                    "prcv_incremental" if args.incremental_prcv else "all_five_methods",
                    Path(report.files["json"]),
                    Path(report.files["config"]),
                    elapsed,
                )
            ],
        }
        if args.incremental_prcv:
            record = merge_prcv_record(prior[identifier], record)
            paths_out = {**prior_paths[identifier], **paths_out}
        records.append(record)
        all_output_paths[identifier] = paths_out
        (args.output / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"{identifier}: executed {', '.join(methods)} in {elapsed:.2f}s", flush=True)
    evidence = {
        "scope": f"{len(records)} selected examples, not a benchmark; "
        "training/test independence unknown",
        "inference": "Original source resolution, no input resize; evaluation mode. "
        "Actual device is recorded per method and run.",
        "icig_input": "Original supplied VIS_Y for ICIG; RGB for color reconstruction",
        "letters_input": "RGB visible, grayscale IR; all pixels valid, material from visible color",
        "prcv_input": "RGB visible and original grayscale IR; supplied PRCV inference adapter",
        "metrics": "Saved gray output: ICIG raw fused gray; "
        "letters and PRCV Pillow L of final color",
        "timing": "Separate earlier four-method and incremental PRCV runs; "
        "not a contemporaneous speed benchmark."
        if args.incremental_prcv
        else "Five methods executed in each recorded workflow invocation.",
        "display": "Precomputed color previews with display scaling and center crops; "
        "not browser neural inference. Metrics use full-resolution gray outputs.",
        "pairs": records,
    }
    if args.incremental_prcv:
        evidence["incremental_provenance"] = {
            "previous_manifest_sha256": sha256(args.previous_manifest),
            "previous_summary_sha256": sha256(args.previous_output / "summary.json"),
        }
    if args.assets:
        prepared = args.output / "prepared-assets"
        for record in records:
            identifier = record["id"]
            record["preview"] = f"{identifier}.png"
            paths = input_paths[identifier]
            preview(
                [
                    paths["visible"],
                    paths["infrared"],
                    *(all_output_paths[identifier][f"{key}_color"] for key in RESEARCH_METHODS),
                ],
                ["Visible", "Infrared", "ICIG (color)", "letters (color)", "PRCV (color)"],
                prepared / record["preview"],
            )
        (prepared / "manifest.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        args.assets.mkdir(parents=True, exist_ok=True)
        for record in records:
            shutil.copy2(prepared / record["preview"], args.assets / record["preview"])
        shutil.copy2(prepared / "manifest.json", args.assets / "manifest.json")
    (args.output / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
