#!/usr/bin/env bash
# Download the training-start and league checkpoints (release "lineage-v1") into checkpoints/lineage/ and verify them
# against checkpoints/lineage.SHA256SUMS (committed in this repository). About 105 MB. Only scripts/train.sh needs them.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/harrison-barnes/metapodV0-pokemon-bot/releases/download/lineage-v1
mkdir -p checkpoints/lineage
for f in $(awk '{print $2}' checkpoints/lineage.SHA256SUMS); do
  [ -f "checkpoints/lineage/$f" ] || echo "downloading $f"; curl -fsSL --retry 3 -o "checkpoints/lineage/$f" "$URL/$f"
done
cd checkpoints/lineage
if command -v sha256sum > /dev/null; then sha256sum -c ../lineage.SHA256SUMS; else shasum -a 256 -c ../lineage.SHA256SUMS; fi
