#!/bin/bash
# Contact labels of the published clips, the clips cleaned against their terrain, and the labels of the cleaned clips.
#   tools/download_data.sh && scripts/prepare_data.sh
# Writes data/php_parkour_contacts, data/php_parkour_refined and data/php_parkour_refined_contacts (about 10 minutes on 16 CPU workers).
# REFINE_ARGS adds flags to refine_motion, e.g. REFINE_ARGS="--clips run_forward_0.6s_step0.8s" refines only the clips whose name contains that.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY -m proacp.clips.labels                                          # data/php_parkour -> data/php_parkour_contacts
$PY -m proacp.clips.refine --out data/php_parkour_refined ${REFINE_ARGS:-}   # whole-trajectory IK: no penetration, contacts on the surface
$PY -m proacp.clips.labels --motion_dir data/php_parkour_refined    # -> data/php_parkour_refined_contacts
