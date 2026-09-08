---
name: dataset-audit
description: Audit paired infrared-visible data after input changes and before dataset experiments.
---

From the repository root, in its Python environment:

1. Run `python .agents/skills/dataset-audit/scripts/audit.py --root examples --output outputs/audit.json`.
2. Review missing counterparts, duplicate stems, format/channel/decode errors, sizes and blank warnings. Dataset mode expects visible/ and infrared/ with matching stems. Only report issues; never delete, overwrite or upload sources. Output the JSON report path, pair counts, issues and suggested fixes.
3. Summarize actual observations and any unresolved failures.
