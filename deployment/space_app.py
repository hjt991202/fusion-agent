"""Public Space entrypoint: deterministic processing only, never paid model calls."""

import os
from pathlib import Path

os.environ["FUSION_PUBLIC_DEMO"] = "1"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "16000000"
os.environ["FUSION_EXAMPLES_DIR"] = str(Path(__file__).resolve().parent / "examples")

from fusion_agent.ui.gradio_app import main  # noqa: E402

if __name__ == "__main__":
    main()
