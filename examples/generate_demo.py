"""Generate original synthetic shapes; these are not real infrared sensor data."""

from pathlib import Path

import cv2
import numpy as np

from fusion_agent.tools.image_io import save_image


def main() -> None:
    directory = Path(__file__).parent
    y, x = np.mgrid[:192, :256]
    visible = (40 + 100 * x / 255 + 25 * np.sin(x / 4) * np.cos(y / 7)).clip(0, 255)
    infrared = 30 + 180 * np.exp(-((x - 160) ** 2 + (y - 95) ** 2) / 600)
    visible = visible.astype(np.uint8)
    cv2.rectangle(visible, (35, 40), (100, 150), 210, 2)
    cv2.putText(visible, "FUSION", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 230, 1)
    save_image(directory / "visible.png", visible)
    save_image(directory / "infrared.png", infrared.astype(np.uint8))


if __name__ == "__main__":
    main()
