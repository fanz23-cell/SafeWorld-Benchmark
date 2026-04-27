#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ROOT_DIR}/.conda/safewm310"
CACHE_DIR="${ROOT_DIR}/.cache"

mkdir -p "${CACHE_DIR}"

CONDA_NO_PLUGINS=true \
CONDA_SOLVER=classic \
XDG_CACHE_HOME="${CACHE_DIR}" \
conda run -p "${ENV_PREFIX}" python "${ROOT_DIR}/run_safety_envs.py"
