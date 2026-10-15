"""Optional SparseGrFormer v3 inference from the supplied ICIGCode archive."""

from .inference import fuse, fuse_with_model, load_model

__all__ = ["fuse", "fuse_with_model", "load_model"]
