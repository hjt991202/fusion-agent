"""Import three attributed CC BY 4.0 TNO pairs from the verified official archive."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from fusion_agent.tools.image_io import inspect_image_pair, save_image

ARCHIVE_MD5 = "1ab9ce3b84dd3e5894630c13abd20953"
PAIRS = {
    "lake": ("lake/VIS_lake_r.bmp", "lake/IR_lake_g.bmp"),
    "helicopter": ("helicopter/VIS_helib_011.bmp", "helicopter/IR_helib_011.bmp"),
    "house": (
        "2_men_in_front_of_house/VIS_meting003_r.bmp",
        "2_men_in_front_of_house/IR_meting003_g.bmp",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("examples/tno"))
    args = parser.parse_args()
    archive_bytes = args.archive.read_bytes()
    if hashlib.md5(archive_bytes).hexdigest() != ARCHIVE_MD5:
        raise ValueError("Archive MD5 differs from official figshare metadata.")
    manifest = {
        "dataset": "TNO Image Fusion Dataset",
        "author": "Alexander Toet",
        "owner": "TNO, Soesterberg, The Netherlands",
        "doi": "https://doi.org/10.6084/m9.figshare.1008029.v2",
        "download": "https://ndownloader.figshare.com/files/1475454",
        "archive_md5": ARCHIVE_MD5,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "changes": "Lossless BMP-to-PNG re-encoding and descriptive filenames; no pixel edits.",
        "pairs": [],
    }
    with zipfile.ZipFile(args.archive) as archive:
        for name, sources in PAIRS.items():
            entries = {}
            for modality, source in zip(["visible", "infrared"], sources, strict=True):
                member = "TNO_Image_Fusion_Dataset/Athena_images/" + source
                original = archive.read(member)
                image = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_UNCHANGED)
                if image is None or image.dtype != np.uint8:
                    raise ValueError(f"Unsupported source: {member}")
                destination = args.output / modality / f"{name}.png"
                save_image(destination, image)
                decoded = cv2.imdecode(np.frombuffer(destination.read_bytes(), np.uint8), -1)
                np.testing.assert_array_equal(image, decoded)
                entries[modality] = {
                    "file": f"{modality}/{name}.png",
                    "source_member": member,
                    "source_sha256": hashlib.sha256(original).hexdigest(),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            info = inspect_image_pair(
                args.output / entries["visible"]["file"], args.output / entries["infrared"]["file"]
            )
            if not info.same_size:
                raise ValueError(f"Source dimensions differ: {name}")
            manifest["pairs"].append({"name": name, **entries, "inspection": info.model_dump()})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Imported {len(PAIRS)} verified pairs into {args.output}")


if __name__ == "__main__":
    main()
