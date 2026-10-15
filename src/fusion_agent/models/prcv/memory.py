from __future__ import annotations

from types import MethodType

import torch


def _self_forward(self, x):
    chunk = self._prcv_window_chunk_size
    original = self._prcv_unchunked_forward
    if x.shape[0] <= chunk:
        return original(x)
    return torch.cat([original(part) for part in x.split(chunk, dim=0)], dim=0)


def _cross_forward(self, q_in, kv_in):
    chunk = self._prcv_window_chunk_size
    original = self._prcv_unchunked_forward
    if q_in.shape[0] != kv_in.shape[0]:
        raise ValueError("Query and key/value window batches must match.")
    if q_in.shape[0] <= chunk:
        return original(q_in, kv_in)
    return torch.cat(
        [
            original(q, kv)
            for q, kv in zip(q_in.split(chunk, dim=0), kv_in.split(chunk, dim=0), strict=True)
        ],
        dim=0,
    )


def enable_window_chunking(model, chunk_size: int = 4):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 0:
        raise ValueError("chunk_size must be an integer >= 0 (0 disables chunking).")
    for module in model.modules():
        kind = type(module).__name__
        if kind not in {"MultiManifoldAttention", "CrossMultiManifoldAttention"}:
            continue
        if chunk_size == 0:
            if hasattr(module, "_prcv_unchunked_forward"):
                module.forward = module._prcv_unchunked_forward
                del module._prcv_unchunked_forward
                del module._prcv_window_chunk_size
            continue
        if not hasattr(module, "_prcv_unchunked_forward"):
            module._prcv_unchunked_forward = module.forward
            wrapper = _self_forward if kind == "MultiManifoldAttention" else _cross_forward
            module.forward = MethodType(wrapper, module)
        module._prcv_window_chunk_size = chunk_size
    return model
