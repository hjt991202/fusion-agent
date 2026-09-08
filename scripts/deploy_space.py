"""Prepare an allowlisted Space upload; publish only with the matching authenticated owner."""

import argparse
import shutil
from pathlib import Path


def prepare(root: Path, stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=False)
    for directory in ["src", "examples"]:
        shutil.copytree(
            root / directory,
            stage / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
    for name in ["pyproject.toml", "LICENSE"]:
        shutil.copy2(root / name, stage / name)
    shutil.copy2(root / "deployment" / "space_app.py", stage / "app.py")
    shutil.copy2(root / "deployment" / "SPACE_README.md", stage / "README.md")
    (stage / "requirements.txt").write_text(".[ui]\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="hjt991202/fusion-agent")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prepare(root, args.stage)
    print(f"Prepared allowlisted files: {args.stage}")
    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        if api.whoami()["name"] != args.repo.split("/")[0]:
            raise ValueError("Authenticated user does not match the requested Space owner.")
        url = api.create_repo(
            args.repo, repo_type="space", space_sdk="gradio", private=False, exist_ok=True
        )
        result = api.upload_folder(
            repo_id=args.repo,
            repo_type="space",
            folder_path=args.stage,
            commit_message="Deploy verified FusionAgent with TNO examples",
        )
        print(url)
        print(result.commit_url)


if __name__ == "__main__":
    main()
