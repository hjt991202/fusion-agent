---
name: fusion-experiment
description: Run reproducible CPU fusion experiments after fusion algorithm or parameter changes.
---

From the repository root, in its Python environment:

1. Run `python .agents/skills/fusion-experiment/scripts/experiment.py --visible examples/visible.png --infrared examples/infrared.png --output outputs/experiment`.
2. Use --method, --weight and explicit --resize as needed. Inspect generated PNGs, config.json and report.json. Seed is 0; the baselines are deterministic. Output run path, input hashes, versions, configuration, timings and measured metrics. Size matching is not registration. Inspect clipping and odd-size reconstruction.
3. Summarize actual observations and any unresolved failures.
