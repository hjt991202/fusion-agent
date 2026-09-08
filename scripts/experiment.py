"""Use canonical CLI semantics and fixed seeds for repeatable experiments."""

import random

import numpy as np

from fusion_agent.cli import main

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    raise SystemExit(main())
