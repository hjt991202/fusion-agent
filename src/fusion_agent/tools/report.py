"""Reports generated exclusively from validated execution results."""

import html

from fusion_agent.schemas import AnalysisReport, MetricResult


def metric_table(metrics: dict[str, MetricResult]) -> str:
    lines = [
        "| Method | Entropy | Std | Avg gradient | Spatial frequency |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in metrics.items():
        lines.append(
            f"| {method} | "
            + " | ".join(f"{value:.4f}" for value in values.model_dump().values())
            + " |"
        )
    return "\n".join(lines)


def generate_markdown_report(report: AnalysisReport) -> str:
    lines = [
        "# FusionAgent experiment",
        "",
        f"Mode: {report.mode}",
        "",
        "## Task",
        html.escape(report.task),
        "",
        "## Inputs",
    ]
    for info in (report.inputs.visible, report.inputs.infrared):
        lines.append(
            f"- {html.escape(info.name)}: {info.width}x{info.height}, "
            f"{info.channels} channel(s), {info.dtype}; SHA256 `{info.sha256}`"
        )
        lines.extend(f"  - {warning}" for warning in info.warnings)
    lines += [
        "",
        "## Plan",
        html.escape(report.plan.rationale),
        "",
        f"Visible weight: {report.plan.visible_weight}; resize: {report.plan.resize}",
        "",
        "## Metrics",
        metric_table(report.metrics),
        "",
        "## Recommendation",
        report.reason,
        "",
        "## Executed tools",
    ]
    lines.extend(f"- {r.tool} ({r.seconds:.6f} s): {r.details}" for r in report.records)
    lines += ["", "## Limitations", *[f"- {x}" for x in report.limitations], "", "## Files"]
    lines.extend(f"- {key}: `{value}`" for key, value in report.files.items())
    return "\n".join(lines) + "\n"
