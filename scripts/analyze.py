"""Render measured metrics, ties and disagreements from an existing report."""

import argparse
from pathlib import Path

from fusion_agent.schemas import AnalysisReport, MetricResult
from fusion_agent.tools.report import metric_table


def analyze(path: Path) -> str:
    report = AnalysisReport.model_validate_json(path.read_text(encoding="utf-8"))
    lines = [metric_table(report.metrics), "", "Per-metric maxima (not perceptual rankings):"]
    winners = []
    for field in MetricResult.model_fields:
        best = max(getattr(value, field) for value in report.metrics.values())
        names = [name for name, value in report.metrics.items() if getattr(value, field) == best]
        winners.append(tuple(names))
        lines.append(f"- {field}: {', '.join(names)}")
    lines.append("Metric disagreement: " + ("yes" if len(set(winners)) > 1 else "no"))
    lines.append("Evidence: one image pair only; insufficient for general quality claims.")
    lines.append("Noise can increase contrast and detail metrics. Inspect artifacts and targets.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(analyze(args.report))
