"""Strict 8-bit input validation and Unicode-safe image IO."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from fusion_agent.schemas import ImageInfo, PairInfo

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {path.name}")
    if path.suffix.lower() not in SUFFIXES:
        raise ValueError("Unsupported image format. Use PNG, JPEG, BMP or TIFF.")
    if not 0 < path.stat().st_size <= 32 * 1024 * 1024:
        raise ValueError("Image must be nonempty and at most 32 MiB.")
    try:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError("Cannot decode image; it may be damaged.") from exc
    if image is None:
        raise ValueError("Cannot decode image; it may be damaged.")
    if image.dtype != np.uint8:
        raise ValueError(
            "Only 8-bit images are supported; convert with an explicit intensity scale."
        )
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("Expected grayscale or three-channel BGR; alpha images are unsupported.")
    if min(image.shape[:2]) < 2 or image.shape[0] * image.shape[1] > 16_000_000:
        raise ValueError("Image must be at least 2x2 and at most 16 megapixels.")
    return image


def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def inspect_image(path: str | Path) -> ImageInfo:
    image = read_image(path)
    warnings = []
    if np.ptp(image) == 0:
        warnings.append(
            "Constant image: may be blank; inspect visually before interpreting metrics."
        )
    return ImageInfo(
        name=Path(path).name,
        width=image.shape[1],
        height=image.shape[0],
        channels=1 if image.ndim == 2 else image.shape[2],
        dtype=str(image.dtype),
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        warnings=warnings,
    )


def inspect_image_pair(visible: str | Path, infrared: str | Path) -> PairInfo:
    v, i = inspect_image(visible), inspect_image(infrared)
    return PairInfo(visible=v, infrared=i, same_size=(v.width, v.height) == (i.width, i.height))


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed.")
    path.write_bytes(encoded.tobytes())
