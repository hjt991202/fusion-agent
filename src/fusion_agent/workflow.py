"""Authoritative local execution with reproducibility artifacts."""

import importlib.metadata
import json
import logging
import platform
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from fusion_agent.schemas import AnalysisReport, FusionPlan, ToolRecord
from fusion_agent.tools.fusion import wavelet_fusion, weighted_average_fusion
from fusion_agent.tools.image_io import grayscale, inspect_image_pair, read_image, save_image
from fusion_agent.tools.metrics import calculate_metrics, compare_fusion_results
from fusion_agent.tools.registration import resize_or_align_pair
from fusion_agent.tools.report import generate_markdown_report

logger = logging.getLogger(__name__)


def run_workflow(
    visible: str | Path,
    infrared: str | Path,
    output: str | Path,
    task: str = "Compare fusion methods",
    plan: FusionPlan | None = None,
    mode: str = "deterministic",
    initial_records: list[ToolRecord] | None = None,
) -> AnalysisReport:
    plan = plan or FusionPlan()
    records = list(initial_records or [])

    def call(name: str, fn: Callable, *args, details: str = "Completed", **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        records.append(ToolRecord(tool=name, seconds=time.perf_counter() - start, details=details))
        logger.info("Completed tool %s", name)
        return result

    pair = call("inspect_image_pair", inspect_image_pair, visible, infrared)
    v, i = grayscale(read_image(visible)), grayscale(read_image(infrared))
    v, i = call(
        "resize_or_align_pair",
        resize_or_align_pair,
        v,
        i,
        resize=plan.resize,
        details="Bilinear infrared resize" if not pair.same_size else "No resize required",
    )
    # Unique directories prevent accidental overwrites and concurrent UI collisions.
    directory = Path(output) / f"run-{uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    metrics, files = {}, {}
    for method in dict.fromkeys(plan.methods):
        if method == "weighted":
            fused = call(
                "weighted_average_fusion",
                weighted_average_fusion,
                v,
                i,
                visible_weight=plan.visible_weight,
            )
        else:
            fused = call("wavelet_fusion", wavelet_fusion, v, i)
        path = directory / f"{method}.png"
        call("save_image", save_image, path, fused, details=path.name)
        files[method] = str(path.resolve())
        metrics[method] = call("calculate_metrics", calculate_metrics, fused, details=method)
    winner, reason = call("compare_fusion_results", compare_fusion_results, metrics)
    files.update(
        {
            name: str((directory / filename).resolve())
            for name, filename in [
                ("report", "report.md"),
                ("json", "report.json"),
                ("config", "config.json"),
            ]
        }
    )
    report = AnalysisReport(
        task=task,
        mode=mode,
        inputs=pair,
        plan=plan,
        metrics=metrics,
        recommendation=winner,
        reason=reason,
        records=records,
        files=files,
        limitations=[
            "Grayscale, 8-bit CPU baselines; no learned model or color reconstruction.",
            "Size matching is not geometric registration; use registered source pairs.",
            "Noise can raise every reported metric. Inspect images and downstream tasks.",
            "No ground-truth reference metrics or dataset-wide quality claims.",
        ],
    )
    start = time.perf_counter()
    generate_markdown_report(report)
    report.records.append(
        ToolRecord(
            tool="generate_markdown_report",
            seconds=time.perf_counter() - start,
            details="Rendered from measured results",
        )
    )
    Path(files["report"]).write_text(generate_markdown_report(report), encoding="utf-8")
    Path(files["json"]).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    versions = {
        name: importlib.metadata.version(name)
        for name in ["fusion-agent", "numpy", "opencv-python-headless", "PyWavelets", "pydantic"]
    }
    config = {
        "plan": plan.model_dump(),
        "seed": 0,
        "randomness": "No stochastic operations",
        "python": platform.python_version(),
        "versions": versions,
        "input_hashes": [pair.visible.sha256, pair.infrared.sha256],
    }
    Path(files["config"]).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return report
