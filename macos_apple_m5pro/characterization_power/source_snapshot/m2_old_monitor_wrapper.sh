#!/bin/zsh
# Adapter for the historical M2 monitor: it only accepts a relative filename.
set -eu
TARGET=${1:?CSV output path required}
TARGET_DIR=${TARGET:h}
TARGET_NAME=${TARGET:t}
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"
exec "${0:A:h}/ANEPowerMonitor" "$TARGET_NAME"
