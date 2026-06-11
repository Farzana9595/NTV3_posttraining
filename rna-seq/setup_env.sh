#!/usr/bin/env bash
# One-shot bootstrap for a fresh SageMaker AI instance.
# Idempotent: safe to re-run. Creates the `ntv3-rnaseq` conda env and installs deps.
#
# Usage:
#   bash setup_env.sh
#
set -euo pipefail

ENV_NAME="ntv3-rnaseq"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

# 1. Make conda usable inside this non-interactive shell
source /opt/conda/etc/profile.d/conda.sh

# 2. Create env if missing (Python 3.12 + tmux from conda-forge)
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] conda env '${ENV_NAME}' already exists — skipping create"
else
  echo "[setup] creating conda env '${ENV_NAME}'"
  conda create -y -n "${ENV_NAME}" -c conda-forge python=3.12 tmux pip
fi

# 3. Install / refresh Python deps
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r "${REQ_FILE}"

# 4. Sanity check
python - <<'PY'
import boto3, requests, openpyxl, pybigtools
print("imports OK:",
      "boto3", boto3.__version__,
      "| requests", requests.__version__,
      "| openpyxl", openpyxl.__version__,
      "| pybigtools", pybigtools.__version__)
PY

tmux -V
echo
echo "[setup] DONE. Activate with:  conda activate ${ENV_NAME}"
