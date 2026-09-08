"""Read-only audit of demo pairs or matching-stem dataset directories."""

import argparse
import json
from pathlib import Path

from fusion_agent.tools.image_io import inspect_image_pair


def audit(root: Path) -> dict:
    issues, pairs = [], []
    if (root / "visible").is_dir() and (root / "infrared").is_dir():
        mappings = []
        for modality in ["visible", "infrared"]:
            mapping = {}
            for path in sorted((root / modality).iterdir()):
                if path.is_file():
                    if path.stem in mapping:
                        issues.append(f"Duplicate stem: {modality}/{path.stem}")
                    mapping[path.stem] = path
            mappings.append(mapping)
        v, i = mappings
        for stem in sorted(v.keys() | i.keys()):
            if stem not in v or stem not in i:
                issues.append(f"Missing counterpart: {stem}")
            else:
                pairs.append((v[stem], i[stem]))
    else:
        pairs = [(root / "visible.png", root / "infrared.png")]
    reports = []
    for visible, infrared in pairs:
        try:
            pair = inspect_image_pair(visible, infrared)
            reports.append(pair.model_dump())
            if not pair.same_size:
                issues.append(f"Size mismatch: {visible.name}")
            for item in [pair.visible, pair.infrared]:
                issues.extend(f"{item.name}: {warning}" for warning in item.warnings)
        except (ValueError, OSError) as exc:
            issues.append(f"{visible.name}: {exc}")
    if not pairs:
        issues.append("No pairs found")
    return {"pairs_checked": len(pairs), "pairs": reports, "issues": issues}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("examples"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"pairs_checked": result["pairs_checked"], "issues": result["issues"]}))
    return int(bool(result["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
