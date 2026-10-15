"""Command-line entry point."""

import argparse
import logging

from fusion_agent.models.runtime import ModelOptions
from fusion_agent.schemas import FusionPlan, selected_methods
from fusion_agent.workflow import run_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Infrared-visible baselines and research fusion")
    parser.add_argument("--visible", required=True)
    parser.add_argument("--infrared", required=True)
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument(
        "--method",
        choices=["weighted", "wavelet", "both", "icig", "letters", "research", "all"],
        default="both",
    )
    parser.add_argument(
        "--weight", type=float, default=0.5, help="Visible weight for weighted fusion"
    )
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--task", default="Compare the selected fusion methods")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--icig-weights")
    parser.add_argument("--letters-weights")
    parser.add_argument("--visible-y", help="Optional prepared ICIG visible luminance")
    parser.add_argument("--valid-mask", help="Letters observed registration support; nonzero=valid")
    parser.add_argument("--material-mask", help="Letters observed material silhouette")
    parser.add_argument("--material-mode", choices=["visible", "valid"], default="visible")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        plan = FusionPlan(
            methods=selected_methods(args.method),
            visible_weight=args.weight,
            resize=args.resize,
        )
        options = ModelOptions(**{name: getattr(args, name) for name in ModelOptions.model_fields})
        if args.agent:
            from fusion_agent.agents.coordinator import run_agent_workflow

            report = run_agent_workflow(
                args.visible, args.infrared, args.output, args.task, plan, model_options=options
            )
        else:
            report = run_workflow(
                args.visible, args.infrared, args.output, args.task, plan, model_options=options
            )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Fusion failed: {exc}\n")
    print(report.files["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
