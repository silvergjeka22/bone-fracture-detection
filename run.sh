#!/usr/bin/env bash
# macOS / Linux / Windows Git Bash:  ./run.sh push [--quick] | status | get | slides   (Windows PowerShell: run.bat)
set -euo pipefail
cd "$(dirname "$0")"
for py in "${PYTHON:-}" python3 python py; do
  if [ -n "$py" ] && "$py" -c "import sys; assert sys.version_info >= (3, 9)" >/dev/null 2>&1; then
    exec "$py" scripts/kaggle_run.py "$@"
  fi
done
echo "Python 3.9+ not found: install Python (python.org) or activate your conda environment." >&2
exit 127
