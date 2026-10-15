"""Audit local Jiandu inputs and copy a deterministic, byte-identical demo subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

MODALITIES = {
    "visible": ("VIS_RGB", "VIS", 3),
    "infrared": ("IR", "IR", 1),
    "visible_y": ("VIS_Y", "VIS", 1),
}
PIXEL_HASH = "SHA-256 of contiguous uint8 decoded pixels, row-major, RGB channel order for color."
SELECTION = (
    "Deterministic farthest-point coverage of min-max-normalized log(pixel area), "
    "log(height/width), visible-Y mean and infrared mean. Start nearest the feature median; "
    "break ties by numeric source ID. No algorithm outputs or quality scores are used."
)


def image_record(path: Path, source_root: Path, channels: int) -> tuple[dict, list[str]]:
    """Read and hash one source without modifying it; preserve original encoded bytes."""
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image could not be decoded")
    actual_channels = 1 if image.ndim == 2 else image.shape[2]
    if actual_channels != channels or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8/{channels} channels, got {image.dtype}/{actual_channels}")
    if path.suffix.lower() != ".png" or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Expected PNG file and PNG signature")
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if channels == 3 else image
    height, width = image.shape[:2]
    warnings = ["Constant image"] if int(image.min()) == int(image.max()) else []
    return {
        "source_file": f"{source_root.name}/{path.relative_to(source_root).as_posix()}",
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "pixel_sha256": hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest(),
        "bytes": len(payload),
        "width": width,
        "height": height,
        "channels": channels,
        "mode": "RGB" if channels == 3 else "L",
        "dtype": "uint8",
        "mean": float(image.mean()),
        "standard_deviation": float(image.std()),
        "minimum": int(image.min()),
        "maximum": int(image.max()),
    }, warnings


def audit_sources(source_root: Path) -> dict:
    """Pair IR<number>/VIS<number> files and audit all three input modalities."""
    mappings: dict[str, dict[int, Path]] = {}
    issues: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    duplicates: dict[tuple[str, str], list[int]] = defaultdict(list)
    for modality, (folder, prefix, _) in MODALITIES.items():
        mapping: dict[int, Path] = {}
        directory = source_root / folder
        if not directory.is_dir():
            raise ValueError(f"Missing input directory: {directory}")
        files = sorted(directory.iterdir())
        counts[folder] = len(files)
        for path in files:
            match = re.fullmatch(rf"{prefix}(\d+)\.png", path.name, flags=re.IGNORECASE)
            if not path.is_file() or match is None:
                issues.append(f"Unexpected input: {folder}/{path.name}")
                continue
            source_id = int(match[1])
            if source_id in mapping:
                issues.append(f"Duplicate numeric ID: {folder}/{source_id}")
                continue
            mapping[source_id] = path
        mappings[modality] = mapping
    source_ids = sorted(set().union(*(mapping.keys() for mapping in mappings.values())))
    records = []
    for source_id in source_ids:
        pair: dict = {"source_id": source_id}
        valid = True
        for modality, (_, _, channels) in MODALITIES.items():
            path = mappings[modality].get(source_id)
            if path is None:
                issues.append(f"Missing counterpart: {modality}/{source_id}")
                valid = False
                continue
            try:
                record, file_warnings = image_record(path, source_root, channels)
                pair[modality] = record
                duplicates[(modality, record["pixel_sha256"])].append(source_id)
                warnings.extend(f"{record['source_file']}: {item}" for item in file_warnings)
                valid = valid and not file_warnings
            except (ValueError, OSError) as exc:
                issues.append(f"{path.relative_to(source_root).as_posix()}: {exc}")
                valid = False
        if all(modality in pair for modality in MODALITIES):
            sizes = {(pair[key]["width"], pair[key]["height"]) for key in MODALITIES}
            if len(sizes) != 1:
                issues.append(f"Size mismatch: {source_id}")
                valid = False
        pair["eligible"] = valid
        records.append(pair)
    duplicate_pixels = [
        {"modality": modality, "pixel_sha256": pixel_hash, "source_ids": ids}
        for (modality, pixel_hash), ids in sorted(duplicates.items())
        if len(ids) > 1
    ]
    return {
        "source": source_root.name,
        "pairing": "Numeric suffix in IR<number>.png and VIS<number>.png; all three modalities",
        "pixel_hash_definition": PIXEL_HASH,
        "file_counts": counts,
        "source_ids_checked": len(source_ids),
        "eligible_pairs": sum(record["eligible"] for record in records),
        "total_bytes": sum(
            record[key]["bytes"] for record in records for key in MODALITIES if key in record
        ),
        "issues": issues,
        "warnings": warnings,
        "duplicate_pixels": duplicate_pixels,
        "pairs": records,
    }


def select_pairs(records: list[dict], count: int) -> list[dict]:
    """Cover size, aspect and input brightness reproducibly without outcome-based selection."""
    candidates = sorted(
        (record for record in records if record["eligible"]), key=lambda item: item["source_id"]
    )
    if count < 1 or count > len(candidates):
        raise ValueError(f"Requested {count} pairs; available eligible pairs: {len(candidates)}")
    features = np.array(
        [
            [
                math.log(item["visible"]["width"] * item["visible"]["height"]),
                math.log(item["visible"]["height"] / item["visible"]["width"]),
                item["visible_y"]["mean"],
                item["infrared"]["mean"],
            ]
            for item in candidates
        ],
        dtype=np.float64,
    )
    spans = np.ptp(features, axis=0)
    features = (features - features.min(axis=0)) / np.where(spans > 0, spans, 1)
    selected = [int(np.argmin(np.sum((features - np.median(features, axis=0)) ** 2, axis=1)))]
    while len(selected) < count:
        distances = np.min(
            np.sum((features[:, None, :] - features[selected][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return sorted((candidates[index] for index in selected), key=lambda item: item["source_id"])


def write_contact_sheet(pairs: list[dict], destination: Path, output: Path) -> None:
    """Create an audit-only contact sheet, with original aspect ratio retained."""
    cell_width, cell_height, label_height = 180, 290, 30
    sheet = np.full((len(pairs) * cell_height, 3 * cell_width, 3), 245, dtype=np.uint8)
    for row, pair in enumerate(pairs):
        for column, modality in enumerate(MODALITIES):
            payload = (destination / pair[modality]["file"]).read_bytes()
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            scale = min((cell_width - 12) / width, (cell_height - label_height - 12) / height)
            thumbnail = cv2.resize(
                image, (max(1, round(width * scale)), max(1, round(height * scale)))
            )
            left = column * cell_width + (cell_width - thumbnail.shape[1]) // 2
            top = row * cell_height + label_height
            sheet[top : top + thumbnail.shape[0], left : left + thumbnail.shape[1]] = thumbnail
            label = f"{pair['source_id']} {modality}"
            cv2.putText(
                sheet,
                label,
                (column * cell_width + 5, row * cell_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
    success, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Contact sheet encoding failed")
    output.write_bytes(encoded.tobytes())


def prepare(source_root: Path, destination: Path, audit_dir: Path, count: int) -> dict:
    """Audit, select and copy only inputs authorized for this repository demonstration."""
    source_root = source_root.resolve()
    destination = destination.resolve()
    audit_dir = audit_dir.resolve()
    for target in [destination, audit_dir]:
        if target == source_root or source_root in target.parents or target in source_root.parents:
            raise ValueError("Outputs must be separate from the source dataset")
    report = audit_sources(source_root)
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "source-audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    selected = select_pairs(report["pairs"], count)
    pairs = []
    for source_pair in selected:
        pair_id = f"jiandu_{source_pair['source_id']:04d}"
        pair = {"id": pair_id, "name": pair_id, "source_id": source_pair["source_id"]}
        for modality in MODALITIES:
            record = dict(source_pair[modality])
            relative = f"{modality}/{pair_id}.png"
            target = destination / relative
            source = source_root.parent / record["source_file"]
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"Source changed during preparation: {record['source_file']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != payload:
                    raise ValueError(f"Refusing to overwrite a different file: {target}")
            else:
                target.write_bytes(payload)
            record["file"] = relative
            pair[modality] = record
        pairs.append(pair)
    manifest = {
        "dataset": "User-provided Jiandu fusion demonstration subset",
        "source": "Local test/IR, test/VIS_RGB and test/VIS_Y supplied by the repository owner",
        "provenance": "The user describes these as their Jiandu fusion research inputs; "
        "ownership and original acquisition details have not been independently verified.",
        "permission": "The user explicitly requested publishing about a dozen test pairs in "
        "their GitHub project for demonstration. This records that request only; it does not "
        "grant a general redistribution license to other parties.",
        "license": "No separate dataset license supplied; not covered by repository MIT license",
        "changes": "Encoded PNG bytes copied unchanged; filenames normalized for matching stems. "
        "No resizing, registration, color conversion or pixel edits.",
        "pixel_hash_definition": PIXEL_HASH,
        "visible_y": "Original supplied VIS_Y images; not recomputed from RGB. The source "
        "luminance conversion is not independently established.",
        "selection": SELECTION,
        "source_pairs_checked": report["source_ids_checked"],
        "source_eligible_pairs": report["eligible_pairs"],
        "pair_count": len(pairs),
        "total_image_bytes": sum(pair[key]["bytes"] for pair in pairs for key in MODALITIES),
        "usage": "Demonstration only. This small, selected subset is not a benchmark or evidence "
        "of model generalization, and its training/test independence is not established.",
        "pairs": pairs,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_contact_sheet(pairs, destination, audit_dir / "selected-contact-sheet.jpg")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("test"))
    parser.add_argument("--destination", type=Path, default=Path("examples/jiandu"))
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/jiandu-audit"))
    parser.add_argument("--count", type=int, default=15)
    args = parser.parse_args()
    manifest = prepare(args.source, args.destination, args.audit_dir, args.count)
    print(json.dumps({key: manifest[key] for key in ["pair_count", "total_image_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
