"""Environment setup, called once at the top of the notebook: Kaggle, Colab or a local machine.

On Kaggle (see kernel-metadata.json and run.sh):
  - the dataset is attached as an input and mounted read-only under /kaggle/input;
  - /kaggle/working is the only folder saved as the kernel's output, so results go there
    (/kaggle/working/results) and the image cache goes to /tmp (it is rebuilt in ~1 minute
    and would only bloat the download);
  - torch, numpy, sklearn, matplotlib are preinstalled: only kymatio, captum and ultralytics (YOLO) are added.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"kymatio": "kymatio>=0.3", "captum": "captum>=0.7", "ultralytics": "ultralytics>=8.3"}


def environment():
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").is_dir():
        return "kaggle"
    if "google.colab" in sys.modules:
        return "colab"
    return "local"


def install():
    """pip-install the packages that are missing (never reinstalls torch: Kaggle's is CUDA-matched)."""
    missing = [spec for name, spec in PACKAGES.items() if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    print("packages ready" + (f" (installed {', '.join(missing)})" if missing else ""))


def setup(root=REPO_ROOT, data_dir=None, install_deps=True):
    """Install missing packages, find the dataset and create the output folders.

    Returns {"env", "root", "data", "results", "figures", "cache"} (paths as pathlib.Path).
    `data_dir` overrides the automatic search (Kaggle: /kaggle/input; elsewhere: <repo>/data);
    the environment variables BFD_DATA_DIR and BFD_RESULTS_DIR do the same without editing the notebook.
    """
    env = environment()
    root = Path(root)
    if install_deps:
        install()

    from .data import find_data_dir  # imported after install(): data.py needs torch/torchvision

    data_dir = data_dir or os.environ.get("BFD_DATA_DIR")
    if data_dir is None:
        search = Path("/kaggle/input") if env == "kaggle" else root / "data"
        data_dir = find_data_dir(search)
    else:
        data_dir = find_data_dir(data_dir)
    if env == "kaggle":
        results, cache = Path("/kaggle/working/results"), Path("/tmp/bfd-cache")
    else:
        results, cache = root / "results", root / "cache"
    if os.environ.get("BFD_RESULTS_DIR"):
        results = Path(os.environ["BFD_RESULTS_DIR"])
        cache = results.parent / "cache"
    paths = {"env": env, "root": root, "data": Path(data_dir), "results": results,
             "figures": results / "figures", "cache": cache}
    for key in ("results", "figures", "cache"):
        paths[key].mkdir(parents=True, exist_ok=True)
    for key, value in paths.items():
        print(f"{key:8s} {value}")
    return paths
