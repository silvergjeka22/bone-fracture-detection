#!/usr/bin/env bash
# Run from Git Bash on Windows after `conda activate bone-fracture` and the
# one-time `kaggle auth login --force` setup documented in README.md.
# The Python launcher uploads an allowlisted local-code snapshot as a private
# Kaggle Dataset, attaches it to the notebook, and starts the remote job.
set -euo pipefail
cd "$(dirname "$0")"

# Test interpreters instead of trusting command -v: Windows Store aliases may
# be present but unusable, while the activated Conda environment is valid.
if [ -n "${PYTHON:-}" ]; then
  "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11)'
else
  candidates=()
  if [ -n "${CONDA_PREFIX:-}" ]; then
    candidates+=("$CONDA_PREFIX/python.exe" "$CONDA_PREFIX/bin/python")
  fi
  candidates+=(python python3)
  for candidate in "${candidates[@]}"; do
    if "$candidate" -c 'import sys; assert sys.version_info >= (3, 11)' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
  if [ -z "${PYTHON:-}" ]; then
    echo "Python 3.11+ was not found. Run: conda activate bone-fracture" >&2
    exit 127
  fi
fi

exec "$PYTHON" scripts/kaggle_run.py "$@"
