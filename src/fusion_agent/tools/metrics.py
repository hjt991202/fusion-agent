"""No-reference statistics on 8-bit grayscale; higher is not always better."""

import numpy as np

from fusion_agent.schemas import MetricResult
from fusion_agent.tools.fusion import validate_array


def calculate_metrics(image: np.ndarray) -> MetricResult:
    validate_array(image)
    x = image.astype(np.float64)
    counts = np.bincount(image.ravel(), minlength=256)
    p = counts[counts > 0] / image.size
    dx, dy = np.diff(x, axis=1), np.diff(x, axis=0)
    return MetricResult(
        entropy=max(0.0, float(-np.sum(p * np.log2(p)))),
        standard_deviation=float(x.std()),
        average_gradient=float(np.sqrt((dx[:-1] ** 2 + dy[:, :-1] ** 2) / 2).mean()),
        spatial_frequency=float(np.sqrt(np.mean(dx**2) + np.mean(dy**2))),
    )


def compare_fusion_results(results: dict[str, MetricResult]) -> tuple[str, str]:
    if not results:
        raise ValueError("No fusion results to compare.")
    winner = max(results, key=lambda key: results[key].average_gradient)
    reason = (
        f"{winner} has the highest average gradient among executed methods "
        "(ties use execution order). This is a detail-oriented heuristic, "
        "not proof of perceptual quality or target preservation."
    )
    return winner, reason
