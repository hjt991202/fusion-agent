---
name: result-analysis
description: Analyze measured fusion results, per-metric maxima and disagreements for experiment reports.
---

From the repository root, in its Python environment:

1. Run `python .agents/skills/result-analysis/scripts/analyze.py RUN/report.json`.
2. Replace RUN with an actual run directory. Read the Markdown table and tied maxima; maxima are not quality labels. Explain metric conflicts and noise sensitivity. Inspect actual PNGs and state the one-pair sample scope. Output the table, metric disagreements, limitations and whether evidence supports the conclusion. Never invent data.
3. Summarize actual observations and any unresolved failures.
