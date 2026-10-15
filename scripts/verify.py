"""Run checks and inspect prospective Git content without modifying Git state."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def candidate_audit() -> list[str]:
    root = Path.cwd()
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "ls-files",
            "-co",
            "--exclude-standard",
            "-z",
        ],
        capture_output=True,
        check=True,
    )
    issues = []
    for filename in set(result.stdout.decode("utf-8").split("\0")) - {""}:
        path = root / filename
        if not path.is_file():
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            issues.append(f"Large file: {filename}")
        if path.name == ".env" or path.suffix in {".pt", ".pth", ".ckpt", ".onnx", ".pyc"}:
            issues.append(f"Forbidden artifact: {filename}")
        if any(part in {"outputs", "datasets", ".venv", "__pycache__"} for part in path.parts):
            issues.append(f"Cache/data artifact: {filename}")
        if path.suffix not in {".png", ".jpg"}:
            content = path.read_text(encoding="utf-8", errors="replace")
            if re.search(
                r"(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|hf_[A-Za-z0-9]{25,}|"
                r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)",
                content,
            ):
                issues.append(f"Potential credential: {filename}")
    return issues


def main() -> int:
    commands = [
        ["-m", "ruff", "format", "--check", "."],
        ["-m", "ruff", "check", "."],
        ["-m", "pytest", "-q"],
        ["examples/generate_demo.py"],
        [
            "-m",
            "fusion_agent.cli",
            "--visible",
            "examples/visible.png",
            "--infrared",
            "examples/infrared.png",
            "--output",
            "outputs/verification",
        ],
    ]
    checks = []
    for command in commands:
        result = subprocess.run(
            [sys.executable, *command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        checks.append(
            {
                "command": command,
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr,
            }
        )
    issues = candidate_audit()
    if shutil.which("node"):
        for source in ["web/app.mjs", "web/fusion.mjs", "web/showcase.mjs"]:
            result = subprocess.run(["node", "--check", source], capture_output=True, text=True)
            checks.append(
                {
                    "command": ["node", "--check", source],
                    "passed": result.returncode == 0,
                    "output": result.stdout + result.stderr,
                }
            )
    report = {
        "checks": checks,
        "candidate_issues": issues,
        "limitations": "Heuristic secret audit; offline tests do not prove live API access.",
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for check in checks:
        print(("PASS " if check["passed"] else "FAIL ") + " ".join(check["command"]))
        if not check["passed"]:
            print(check["output"])
    print("Candidate issues:", issues)
    return int(bool(issues) or not all(check["passed"] for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
