# Open-science anonymous artifact copy

This directory contains the anonymized code and models for the paper's
observation-state characterization experiments.

Included contents:

- source code
- scripts
- README files
- configuration files
- the model/workload files marked `bundled` in `models/MODEL_INDEX.tsv`

Path policy:

- Absolute personal paths were rewritten to relative placeholders.
- External assets are expected under the local placeholder folders in this tree,
  such as `external/`, `assets/`, `figures/`, `manuscript/`, `home/`, and
  `windows/`.
- If a script needs a local monitor or runtime dependency, place it under the
  corresponding placeholder path or update the relative path in that script.

Notes:

- This copy preserves the runnable source layout as much as possible without
  retaining personal path information.
