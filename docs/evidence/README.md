# RSL Evidence

This directory contains reproducible evidence for the bounded RSL assignment.

- `rsl-demo-v2.json`: final JSON-format task result after output normalization.
- `rsl-experiment-20261015T114936Z.json`: latest three-task controlled experiment.
- `rsl-evidence-summary.json`: compact summary of metrics, memory promotion, rollback, and the detected failure case.
- `rsl-memory-promotion-v2.json`: two verified runs that promote candidate memory to active memory.
- `rsl-memory-retrieval-rollback-v2.json`: active memory retrieval followed by rollback.
- `rsl-demo-poem-v2.json`: earlier failure case showing that the 1.5B model could not repair multiline format from feedback alone.

All scores come from real local-model API calls. No training-time weight updates are performed.
