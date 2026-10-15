"""Build a self-contained static demo and optionally upload to the user's free Space."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def prepare_static(root: Path, stage: Path) -> None:
    """Stage only the curated examples and measured previews, preserving image notices."""
    jiandu = root / "examples" / "jiandu"
    showcase = root / "docs" / "assets" / "jiandu"
    inputs = json.loads((jiandu / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((showcase / "manifest.json").read_text(encoding="utf-8"))
    selected = {pair["id"] for pair in inputs["pairs"]}
    if (
        len(inputs["pairs"]) != 15
        or len(results["pairs"]) != 15
        or len(selected) != 15
        or {pair["id"] for pair in results["pairs"]} != selected
        or any(re.fullmatch(r"jiandu_\d{4,}", identifier) is None for identifier in selected)
    ):
        raise ValueError("Static showcase must match the 15 curated Jiandu examples")
    by_id = {pair["id"]: pair for pair in inputs["pairs"]}
    for pair in results["pairs"]:
        source = by_id[pair["id"]]
        if (
            pair.get("source_id") != source["source_id"]
            or pair.get("width") != source["visible"]["width"]
            or pair.get("height") != source["visible"]["height"]
            or pair.get("input_hashes")
            != {key: source[key]["sha256"] for key in ["visible", "infrared", "visible_y"]}
        ):
            raise ValueError("Showcase measurements do not match the curated source inputs")
        if pair.get("preview") != f"{pair['id']}.png":
            raise ValueError("Unexpected showcase preview path")
    image_files = []
    for pair in inputs["pairs"]:
        for modality in ["visible", "infrared", "visible_y"]:
            record = pair[modality]
            relative = Path(record["file"])
            if relative.parts != (modality, f"{pair['id']}.png"):
                raise ValueError("Unexpected curated input path")
            path = jiandu / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Curated input hash mismatch")
            image_files.append(relative)
    previews = []
    for pair in results["pairs"]:
        preview = showcase / pair["preview"]
        if not preview.is_file():
            raise ValueError("Missing measured showcase preview")
        previews.append(preview)
    stage.mkdir(parents=True, exist_ok=False)
    for name in ["index.html", "app.mjs", "fusion.mjs", "style.css", "showcase.mjs"]:
        shutil.copy2(root / "web" / name, stage / name)
    shutil.copytree(root / "examples" / "tno", stage / "samples" / "tno")
    for relative in [Path("README.md"), Path("manifest.json"), *image_files]:
        destination = stage / "samples" / "jiandu" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jiandu / relative, destination)
    (stage / "showcase").mkdir()
    for path in [showcase / "manifest.json", *previews]:
        shutil.copy2(path, stage / "showcase" / path.name)
    for name in ["visible.png", "infrared.png"]:
        shutil.copy2(root / "examples" / name, stage / "samples" / name)
    shutil.copy2(root / "LICENSE", stage / "LICENSE")
    (stage / "README.md").write_text(
        "---\ntitle: FusionAgent\nemoji: 🔬\ncolorFrom: blue\ncolorTo: indigo\n"
        "sdk: static\napp_file: index.html\npinned: false\nlicense: mit\n"
        "short_description: Jiandu fusion showcase and browser baselines\n---\n\n"
        "# FusionAgent · 在线图像融合\n\n"
        "Select a sample pair and run both fusion baselines in your browser. "
        "Uploaded files never leave your browser. No backend or API key required. "
        "ICIG and letters results for 15 Jiandu pairs were computed in Python beforehand; "
        "the browser does not execute these neural models.\n\n"
        "Source: https://github.com/hjt991202/fusion-agent\n\n"
        "TNO images: Alexander Toet / TNO, Soesterberg, The Netherlands, CC BY 4.0, "
        "https://doi.org/10.6084/m9.figshare.1008029.v2 . "
        "See samples/tno/README.md and manifest.json for attribution and provenance. "
        "The MIT code license does not replace the image license.\n\n"
        "Jiandu images: supplied by the project owner for this project demonstration. "
        "No general dataset redistribution license was supplied; these images and their "
        "derived previews are not covered by MIT or TNO's CC BY license. "
        "See samples/jiandu/README.md and manifest.json for scope and source hashes. "
        "Measured results: showcase/manifest.json. The selected subset is not a benchmark; "
        "training/test independence has not been established.\n\n"
        "Browser grayscale/decoding may differ from OpenCV; Haar roundoff can differ by "
        "one intensity level. Live browser computation runs only weighted/Haar baselines, "
        "without Agent/model calls or registration.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--repo", default="hjt991202/fusion-agent")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    prepare_static(Path(__file__).resolve().parents[1], args.stage)
    print(f"Prepared static demo: {args.stage}")
    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        if api.whoami()["name"] != args.repo.split("/")[0]:
            raise ValueError("Authenticated owner mismatch.")
        url = api.create_repo(
            args.repo, repo_type="space", space_sdk="static", private=False, exist_ok=True
        )
        info = api.repo_info(args.repo, repo_type="space")
        if info.sdk != "static":
            raise ValueError("Refusing to overwrite a non-static Space.")
        commit = api.upload_folder(
            repo_id=args.repo,
            repo_type="space",
            folder_path=args.stage,
            commit_message="Publish verified Jiandu showcase and browser fusion baselines",
        )
        print(url)
        print(commit.commit_url)


if __name__ == "__main__":
    main()
