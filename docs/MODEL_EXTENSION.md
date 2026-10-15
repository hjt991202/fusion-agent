# Optional research model integration

The original weighted/Haar `FusionMethod` protocol remains a grayscale baseline contract.
ICIG and letters live in separate optional packages under `src/fusion_agent/models/` because
their original RGB, masks, tiling and refinement behavior requires a richer interface.

`fuse()` accepts uint8 image arrays and explicit local checkpoint/device options, and returns
`gray`, optional RGB `color`, and JSON-safe execution metadata. The workflow converts its
OpenCV BGR inputs to RGB at this boundary and converts color results back to BGR when saving.
It computes metrics on the saved grayscale array and records checkpoint/config/input hashes.

`ModelOptions` is separate from `FusionPlan`: the Agent can select only user-allowed methods
and baseline parameters, and cannot change bound checkpoint paths, masks, devices or thread limits.
No model import or checkpoint load occurs for baseline-only runs. Checkpoints are loaded with
`weights_only=True`, strict state matching, evaluation mode and disabled gradients.

The adapters preserve the supplied inference mathematics rather than forcing every network to
use a common [0,1] input convention. ICIG takes [0,255] grayscale tiles; letters retains its own
normalization, learned flow and refinement. See [details and commands](RESEARCH_METHODS.md).

Further work: validated alignment accuracy, training/test independence, reference/downstream
metrics and optional GPU validation. None is implied by local CPU execution or visual inspection.
