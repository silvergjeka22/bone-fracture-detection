#!/usr/bin/env bash
# Run the notebook on a Kaggle GPU kernel from your terminal, then fetch the results.
#
# One-time setup (see README, "Run it on Kaggle"):
#   1) pip install kaggle, then `kaggle auth login` (or a token in ~/.kaggle/access_token or kaggle.json)
#   2) on kaggle.com add a Secret GITHUB_TOKEN (a GitHub token that can read this private repo)
#   3) put YOUR Kaggle username in kernel-metadata.json ("id": "<username>/bone-fracture-detection")
#
#   ./run.sh push [--quick]   send the notebook to Kaggle and start it (then you can switch your PC off)
#   ./run.sh status           queued / running / complete / error
#   ./run.sh get              download the output into ./out and copy the results into ./results
#   ./run.sh slides           rebuild presentation/main.pdf from ./results
#   ./run.sh stop             open the kernel page (Kaggle has no CLI stop)
#
# BRANCH=<git branch> ./run.sh push   makes the kernel clone that branch (default: main).
# On Windows, run this from Git Bash after `conda activate bone-fracture`.
set -e
cd "$(dirname "$0")"

# Conda on Windows normally exposes `python`, while Linux/macOS installations
# often expose `python3`. Prefer `python` to avoid Windows' Store `python3`
# alias, which is not the interpreter from the active Conda environment.
if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "Python was not found. Activate the Conda environment, then retry."
  exit 127
fi

KERNEL=$($PYTHON -c "import json; print(json.load(open('kernel-metadata.json'))['id'])")
BRANCH="${BRANCH:-main}"

case "${1:-help}" in
  push)
    # Push a copy of the notebook with BRANCH (and QUICK for --quick) filled in; the repo copy is untouched.
    QUICK=False; [ "${2:-}" = "--quick" ] && QUICK=True
    STAGE=$(mktemp -d)
    $PYTHON - "$STAGE" "$BRANCH" "$QUICK" <<'EOF'
import json, re, sys
stage, branch, quick = sys.argv[1:]
nb = json.load(open("notebooks/main.ipynb"))
for cell in nb["cells"]:
    src = "".join(cell["source"])
    src = re.sub(r'(REPO, BRANCH = "silvergjeka22/bone-fracture-detection", ")[^"]*(")',
                 rf'\g<1>{branch}\g<2>', src)
    src = src.replace("QUICK      = False", f"QUICK      = {quick}")
    cell["source"] = src
json.dump(nb, open(f"{stage}/main.ipynb", "w"), indent=1)
meta = json.load(open("kernel-metadata.json"))
meta["code_file"] = "main.ipynb"
json.dump(meta, open(f"{stage}/kernel-metadata.json", "w"), indent=2)
print(f"pushing branch '{branch}', QUICK={quick}")
EOF
    kaggle kernels push -p "$STAGE"
    rm -rf "$STAGE" ;;

  status)
    kaggle kernels status "$KERNEL" ;;

  get)
    rm -rf ./out && mkdir -p ./out
    kaggle kernels output "$KERNEL" -p ./out
    if [ -d ./out/results ]; then
      mkdir -p ./results && cp -r ./out/results/. ./results/
      echo "results copied into ./results (open the notebook with TRAIN = False, or run ./run.sh slides)"
    fi ;;

  slides)
    make -C presentation ;;

  stop)
    URL="https://www.kaggle.com/code/${KERNEL}"
    echo "Kaggle cannot stop a kernel from the CLI. Open $URL and click 'Stop Session'."
    { command -v open >/dev/null && open "$URL"; } || { command -v xdg-open >/dev/null && xdg-open "$URL"; } || true ;;

  *)
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
