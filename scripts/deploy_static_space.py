"""Build a self-contained static demo and optionally upload to the user's free Space."""

import argparse
import shutil
from pathlib import Path


def prepare_static(root: Path, stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=False)
    for name in ["index.html", "app.mjs", "fusion.mjs", "style.css"]:
        shutil.copy2(root / "web" / name, stage / name)
    shutil.copytree(root / "examples" / "tno", stage / "samples" / "tno")
    for name in ["visible.png", "infrared.png"]:
        shutil.copy2(root / "examples" / name, stage / "samples" / name)
    shutil.copy2(root / "LICENSE", stage / "LICENSE")
    (stage / "README.md").write_text(
        "---\ntitle: FusionAgent\nemoji: 🔬\ncolorFrom: blue\ncolorTo: indigo\n"
        "sdk: static\napp_file: index.html\npinned: false\nlicense: mit\n"
        "short_description: Browser image fusion with public TNO examples\n---\n\n"
        "# FusionAgent · 在线图像融合\n\n"
        "Select a sample pair and run both fusion baselines in your browser. "
        "Uploaded files never leave your browser. No backend or API key required.\n\n"
        "Source: https://github.com/hjt991202/fusion-agent\n\n"
        "TNO images: Alexander Toet / TNO, Soesterberg, The Netherlands, CC BY 4.0, "
        "https://doi.org/10.6084/m9.figshare.1008029.v2 . "
        "See samples/tno/README.md and manifest.json for attribution and provenance. "
        "The MIT code license does not replace the image license.\n\n"
        "Browser grayscale/decoding may differ from OpenCV; Haar roundoff can differ by "
        "one intensity level. This demo does not run Agent/model calls or registration.\n",
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
            commit_message="Publish verified browser fusion and TNO samples",
        )
        print(url)
        print(commit.commit_url)


if __name__ == "__main__":
    main()
