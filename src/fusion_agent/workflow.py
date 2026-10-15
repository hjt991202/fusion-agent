"""Authoritative local execution with reproducibility artifacts."""

import importlib.metadata
import json
import logging
import platform
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from fusion_agent.models.runtime import ModelOptions, file_evidence, load_adapter
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
    model_options: ModelOptions | None = None,
) -> AnalysisReport:
    plan = plan or FusionPlan()
    model_options = model_options or ModelOptions()
    records = list(initial_records or [])

    def call(name: str, fn: Callable, *args, details: str = "Completed", **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        records.append(ToolRecord(tool=name, seconds=time.perf_counter() - start, details=details))
        logger.info("Completed tool %s", name)
        return result

    pair = call("inspect_image_pair", inspect_image_pair, visible, infrared)
    visible_image, infrared_image = read_image(visible), read_image(infrared)
    v, i = grayscale(visible_image), grayscale(infrared_image)
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
    model_evidence = {}
    for method in dict.fromkeys(plan.methods):
        if method == "weighted":
            fused = call(
                "weighted_average_fusion",
                weighted_average_fusion,
                v,
                i,
                visible_weight=plan.visible_weight,
            )
        elif method == "wavelet":
            fused = call("wavelet_fusion", wavelet_fusion, v, i)
        else:
            adapter = load_adapter(method)
            checkpoint = model_options.checkpoint(method)
            rgb = (
                cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB)
                if visible_image.ndim == 3
                else visible_image
            )
            extra = {}
            auxiliary = {}
            if method == "icig" and model_options.visible_y:
                luminance = read_image(model_options.visible_y)
                if luminance.ndim != 2 or luminance.shape != v.shape:
                    raise ValueError("Prepared visible luminance must be grayscale and match VIS.")
                extra["visible_luminance"] = luminance
                auxiliary["visible_y"] = file_evidence(model_options.visible_y)
            if method == "letters":
                if rgb.ndim == 2:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
                extra["material_mode"] = model_options.material_mode
                for key in ["valid_mask", "material_mask"]:
                    mask_path = getattr(model_options, key)
                    if mask_path:
                        mask_image = read_image(mask_path)
                        if mask_image.ndim != 2 or mask_image.shape != v.shape:
                            raise ValueError("Masks must be grayscale and match the visible grid.")
                        extra[key] = mask_image > 0
                        auxiliary[key] = file_evidence(mask_path)
            model_ir = i
            if method == "letters" and infrared_image.ndim == 3:
                model_ir = cv2.cvtColor(infrared_image, cv2.COLOR_BGR2RGB)
                if not pair.same_size:
                    model_ir = cv2.resize(
                        model_ir, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR
                    )
            result = call(
                f"{method}_fusion",
                adapter.fuse,
                rgb,
                model_ir,
                weights=checkpoint,
                device=model_options.device,
                threads=model_options.threads,
                details=f"Checkpoint SHA256: {file_evidence(checkpoint)['sha256']}",
                **extra,
            )
            fused = result["gray"]
            if fused.shape != v.shape or fused.dtype != np.uint8:
                raise RuntimeError("Model adapter returned an invalid grayscale output.")
            if result.get("color") is not None:
                color_path = directory / f"{method}_color.png"
                call(
                    "save_image",
                    save_image,
                    color_path,
                    cv2.cvtColor(result["color"], cv2.COLOR_RGB2BGR),
                    details=color_path.name,
                )
                files[f"{method}_color"] = str(color_path.resolve())
            model_evidence[method] = {
                "checkpoint": file_evidence(checkpoint),
                "auxiliary_inputs": auxiliary,
                "metadata": result["metadata"],
            }
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
            "Metrics use the saved 8-bit grayscale outputs; color outputs are supplementary.",
            "Size matching is not geometric registration; use registered source pairs.",
            "Noise can raise every reported metric. Inspect images and downstream tasks.",
            "No ground-truth reference metrics or dataset-wide quality claims.",
        ]
        + (
            [
                "Learned inference uses user-supplied checkpoints; no training is performed.",
                "ICIG uses tiled inference and detail/chroma restoration; letters includes learned "
                "flow and boundary refinement, not independently validated registration.",
                "Without a validity mask, letters assumes every input pixel is observed.",
            ]
            if model_evidence
            else []
        ),
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
    if model_evidence:
        versions.update(
            {
                name: importlib.metadata.version(name)
                for name in ["torch", "scipy", "einops", "Pillow"]
            }
        )
    config = {
        "plan": plan.model_dump(),
        "seed": 0,
        "randomness": "Evaluation mode; no stochastic operations in baseline inference. "
        "Learned-model reproducibility is scoped to the recorded software/device.",
        "python": platform.python_version(),
        "versions": versions,
        "input_hashes": [pair.visible.sha256, pair.infrared.sha256],
        "models": model_evidence,
        "preprocessing": {
            "baseline_grayscale": "OpenCV BGR2GRAY",
            "infrared_grayscale": "OpenCV BGR2GRAY (unchanged for grayscale input)",
            "letters_infrared": "Original grayscale or RGB; model retains its source conversion",
            "resize": "infrared bilinear to visible grid" if not pair.same_size else "none",
            "model_options": model_options.model_dump(
                exclude={
                    "icig_weights",
                    "letters_weights",
                    "visible_y",
                    "valid_mask",
                    "material_mask",
                }
            ),
        },
    }
    Path(files["config"]).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return report
