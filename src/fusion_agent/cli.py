"""Command-line entry point."""

import argparse
import logging

from fusion_agent.schemas import FusionPlan
from fusion_agent.workflow import run_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Infrared-visible CPU fusion")
    parser.add_argument("--visible", required=True)
    parser.add_argument("--infrared", required=True)
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument("--method", choices=["weighted", "wavelet", "both"], default="both")
    parser.add_argument(
        "--weight", type=float, default=0.5, help="Visible weight for weighted fusion"
    )
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--task", default="Compare the two fusion baselines")
    parser.add_argument("--agent", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        plan = FusionPlan(
            methods=["weighted", "wavelet"] if args.method == "both" else [args.method],
            visible_weight=args.weight,
            resize=args.resize,
        )
        if args.agent:
            from fusion_agent.agents.coordinator import run_agent_workflow

            report = run_agent_workflow(args.visible, args.infrared, args.output, args.task, plan)
        else:
            report = run_workflow(args.visible, args.infrared, args.output, args.task, plan)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Fusion failed: {exc}\n")
    print(report.files["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
