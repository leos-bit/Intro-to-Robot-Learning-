#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

exec conda run --no-capture-output -n RLPR \
  python -u jax_implementation/MBRL/dynamics/PETS_MPC.py \
  --preset fast \
  --render_completed_run \
  "$@"
