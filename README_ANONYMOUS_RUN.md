# Anonymous run guide

This guide explains how to run the anonymous observation-state artifact.

## Expected local layout

Place any external dependency you need under the placeholder folders shipped in
this tree:

- `external/` for runtime roots and monitor dependencies
- `assets/` for recovered input assets and source snapshots
- `figures/` for manuscript figure targets
- `manuscript/` for manuscript-local source material
- `home/` for home-directory style placeholders
- `windows/` for Windows-specific source placeholders
- `tools/` for helper scripts bundled with the release

## Running code

Each platform directory contains its own README with the preserved command
sequence.

- Linux / macOS code can be run directly once the referenced local dependencies
  exist under the placeholder paths.
## Reproducibility note

The bundled models are indexed in `models/MODEL_INDEX.tsv`. Runtime-specific
dependencies should be restored under the placeholder folders before execution.
