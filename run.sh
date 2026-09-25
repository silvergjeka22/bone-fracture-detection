#!/usr/bin/env bash
# Run the notebook on a Kaggle GPU kernel from your terminal, then fetch the results.
#
# One-time setup (see README, "Run it on Kaggle"):
#   1) pip install kaggle           and put kaggle.json in ~/.kaggle/ (chmod 600)
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
set -e
cd "$(dirname "$0")"
KERNEL=$(python3 -c "import json; print(json.load(open('kernel-metadata.json'))['id'])")
BRANCH="${BRANCH:-main}"

case "${1:-help}" in
  push)
    # Push a copy of the notebook with BRANCH (and QUICK for --quick) filled in; the repo copy is untouched.
    QUICK=False; [ "${2:-}" = "--quick" ] && QUICK=True
    STAGE=$(mktemp -d)
    python3 - "$STAGE" "$BRANCH" "$QUICK" <<'EOF'
import json, sys
stage, branch, quick = sys.argv[1:]
nb = json.load(open("notebooks/main.ipynb"))
for cell in nb["cells"]:
    src = "".join(cell["source"])
    src = src.replace('REPO, BRANCH = "silvergjeka22/bone-fracture-detection", "main"',
                      f'REPO, BRANCH = "silvergjeka22/bone-fracture-detection", "{branch}"')
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
