"""Run the curated source-resolution examples and build measured demonstration previews."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from fusion_agent.models.runtime import ModelOptions
from fusion_agent.schemas import FusionPlan
from fusion_agent.workflow import run_workflow


def preview(paths: list[Path], labels: list[str], target: Path) -> None:
    """Show entire images plus identical center crops; only the display is rescaled."""
    plate = Image.new("RGB", (1000, 740), "#edf1f7")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=Path("examples/jiandu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, help="Optional curated publication-preview folder")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.examples / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for pair in manifest["pairs"]:
        identifier = pair["id"]
        if not identifier.replace("_", "").isalnum():
            raise ValueError("Unsafe example identifier")
        paths = {}
        for modality in ["visible", "infrared", "visible_y"]:
            path = (args.examples / pair[modality]["file"]).resolve()
            if not path.is_relative_to(args.examples.resolve()):
                raise ValueError("Example path must remain inside the selected dataset")
            if hashlib.sha256(path.read_bytes()).hexdigest() != pair[modality]["sha256"]:
                raise ValueError("Example hash does not match the manifest")
            paths[modality] = path
        started = time.perf_counter()
        report = run_workflow(
            paths["visible"],
            paths["infrared"],
            args.output / identifier,
            task=f"Jiandu demonstration {identifier}",
            plan=FusionPlan(
                methods=["weighted", "wavelet", "icig", "letters"],
                rationale="Compare existing baselines and the two supplied methods.",
            ),
            model_options=ModelOptions(
                visible_y=str(paths["visible_y"]),
                device=args.device,
                threads=args.threads,
            ),
        )
        config = json.loads(Path(report.files["config"]).read_text(encoding="utf-8"))
        record = {
            "id": identifier,
            "source_id": pair["source_id"],
            "width": pair["visible"]["width"],
            "height": pair["visible"]["height"],
            "input_hashes": {k: pair[k]["sha256"] for k in paths},
            "metrics": {k: v.model_dump() for k, v in report.metrics.items()},
            "seconds": time.perf_counter() - started,
            "timings": [r.model_dump() for r in report.records if r.tool.endswith("fusion")],
            "models": config["models"],
            "versions": config["versions"],
            "output_hashes": {},
            "clipping": {},
        }
        for key in ["icig", "icig_color", "letters", "letters_color"]:
            path = Path(report.files[key])
            record["output_hashes"][key] = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as im:
                array = np.asarray(im)
                if array.shape[:2] != (record["height"], record["width"]):
                    raise RuntimeError("Output dimensions changed")
                record["clipping"][key] = {
                    "fraction_at_zero": float(np.mean(array == 0)),
                    "fraction_at_255": float(np.mean(array == 255)),
                }
        if args.assets:
            record["preview"] = f"{identifier}.png"
            preview(
                [
                    paths["visible"],
                    paths["infrared"],
                    Path(report.files["icig_color"]),
                    Path(report.files["letters_color"]),
                ],
                ["Visible", "Infrared", "ICIG (color)", "letters (color)"],
                args.assets / record["preview"],
            )
        records.append(record)
        (args.output / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"{identifier}: completed all four methods in {record['seconds']:.2f}s", flush=True)
    if args.assets:
        evidence = {
            "scope": f"{len(records)} selected examples, not a benchmark; "
            "training/test independence unknown",
            "inference": "Original source resolution, no input resize; evaluation mode. "
            "Actual device is recorded per method and pair.",
            "icig_input": "Original supplied VIS_Y for ICIG; RGB for color reconstruction",
            "letters_input": "RGB visible, grayscale IR; all pixels valid, "
            "material from visible color",
            "metrics": "Saved gray output: ICIG raw fused gray; letters Pillow L of final color",
            "display": "Precomputed color previews with display scaling and center crops; "
            "not browser neural inference. Metrics use full-resolution gray outputs.",
            "pairs": records,
        }
        (args.assets / "manifest.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
