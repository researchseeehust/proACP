#!/bin/bash
# Download the data proACP trains on, into data/ (not tracked):
#   data/g1_description/      Unitree G1 URDF + meshes (BSD-3-Clause, via the Hugging Face dataset
#                             lvhaidong/LAFAN1_Retargeting_Dataset) and the MJCF g1_29dof.xml built from it
#   data/php_parkour/motions-*/   the five PHP parkour motion bundles (296 G1 clips with their box terrains),
#                             from the amazon-far/php_parkour GitHub release "training-motion-examples-v1"; each archive
#                             is checked against its published sha256 before it is unpacked
#   tools/download_data.sh              everything
#   tools/download_data.sh --robot      only the G1 description
#   tools/download_data.sh --motions    only the motion bundles
# Existing folders are kept. The motion data is not redistributed in this repository; see the licenses in the README.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
WHAT=${1:-all}

robot() {
  local HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
  local API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main
  list() { curl -sf "$API/$1" | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin)]"; }
  if [ -f data/g1_description/g1_29dof.xml ]; then echo "data/g1_description/g1_29dof.xml exists, skipping the robot"; return; fi
  mkdir -p data/g1_description/meshes
  curl -sfL -o data/g1_description/g1_29dof_rev_1_0.urdf "$HF/resolve/main/robot_description/g1/g1_29dof_rev_1_0.urdf"
  for f in $(list robot_description/g1/meshes); do curl -sfL -o "data/g1_description/meshes/$(basename "$f")" "$HF/resolve/main/$f"; done
  echo "meshes: $(ls data/g1_description/meshes | wc -l)"
  $PY tools/urdf_to_mjcf.py          # builds data/g1_description/g1_29dof.xml (free-floating pelvis)
}

# name sha256 (the archive is <base>/php-<name>-v1.tar.gz)
BASE=https://github.com/amazon-far/php_parkour/releases/download/training-motion-examples-v1
BUNDLES=(
  "motions-locomotion       2397220209b5ce32cd886340f5b12f62821f17e40bff0d2958b1a70295a08dcf"
  "motions-low-step         1f63ee82004d6e705d64b293a9ff6988304ea757394d309209eeabb1b5ee8809"
  "motions-high-step        c4cf035cf2a81336a42277584ef65ee16044f9c5f9d921b08e84c1adaab7381d"
  "motions-low-climb-76     1d94450645e5f6ae457c65425bd06b56851ca3793b1e2d9fdd80bba4821e533a"
  "motions-high-climb-76    045b7ef2ce2231fc8a27aaba16e86d67bb43435d40754df4e0f1ad02ad6e6e05"
)

motions() {
  mkdir -p data/php_parkour data/.download
  local entry name sha tmp
  for entry in "${BUNDLES[@]}"; do
    read -r name sha <<<"$entry"
    if [ -d "data/php_parkour/$name" ]; then echo "data/php_parkour/$name exists, skipping"; continue; fi
    local tgz="data/.download/php-$name-v1.tar.gz"
    curl -sfL --retry 3 -o "$tgz" "$BASE/php-$name-v1.tar.gz"
    echo "$sha  $tgz" | sha256sum -c --quiet - || { echo "sha256 mismatch for $name, not unpacked"; rm -f "$tgz"; exit 1; }
    if tar tzf "$tgz" | grep -qE '^/|(^|/)\.\.(/|$)'; then echo "unsafe paths in $tgz, not unpacked"; exit 1; fi
    tmp=$(mktemp -d data/.download/unpack.XXXXXX)
    tar xzf "$tgz" -C "$tmp" --no-same-owner --no-same-permissions
    mv "$tmp" "data/php_parkour/$name"
    rm -f "$tgz"
    echo "$name: $(ls data/php_parkour/$name | grep -c _motion.npz) clips"
  done
  rmdir data/.download 2>/dev/null || true
}

case "$WHAT" in
  all) robot; motions ;;
  --robot) robot ;;
  --motions) motions ;;
  *) echo "usage: tools/download_data.sh [--robot | --motions]"; exit 1 ;;
esac
