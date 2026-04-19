#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

exec conda run --no-capture-output -n RLPR \
  python -u jax_implementation/MBRL/dynamics/PETS_MPC.py \
  --render \
  --real_time \
  --jit_step \
  "$@"
