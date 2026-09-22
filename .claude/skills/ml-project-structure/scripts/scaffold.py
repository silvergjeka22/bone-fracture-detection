"""Create the skeleton of a simple ML project: flat src/ modules + one thin notebook.

Existing files are never overwritten, so the script can also fill the gaps of an existing project.

    python scaffold.py <project_dir> --title "Bone fracture detection" \
        --modules data models training plots utils \
        --sections "Data" "Models" "Training" "Evaluation" "Discussion"
"""

import argparse
from pathlib import Path

import nbformat as nbf

MODULE_DOCS = {
    "data": "Dataset class, transforms, data loaders and a dataset summary table.",
    "models": "Model classes and build_model(name) -> nn.Module.",
    "training": "Training loop, cross-validation, test metrics, saving/loading weights and results.",
    "plots": "Every figure of the notebook: one function per figure, each with an optional save_to= path.",
    "utils": "Small helpers: set_seed, get_device, count_parameters.",
    "xai": "Explainability methods applied the same way to every model, plus caching of the maps.",
}

UTILS = '''"""Small helpers: reproducibility and device selection."""

import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
'''

GITIGNORE = """__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
kaggle.json
*token*.txt
data/
*.pth
"""

REQUIREMENTS = """torch
torchvision
scikit-learn
numpy
pandas
matplotlib
"""


def write(path, text):
    if path.exists():
        print(f"  keep    {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"  create  {path}")


def notebook(title, modules, sections):
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\nAll code lives in `src/`; this notebook only sets "
                                      "parameters and calls those functions. Run it top to bottom.")]
    cells.append(nbf.v4.new_code_cell(
        "# Setup: works locally (opened from notebooks/) and on Google Colab\n"
        "import sys\nfrom pathlib import Path\n\n"
        'IN_COLAB = "google.colab" in sys.modules\n'
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        '    drive.mount("/content/drive")\n'
        '    PROJECT_ROOT = Path("/content/drive/MyDrive/<project-folder>")\n'
        "else:\n"
        "    PROJECT_ROOT = Path.cwd().parent\n"
        "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        f"from src import {', '.join(modules)}"))
    cells.append(nbf.v4.new_code_cell(
        "# Settings: the only cell to edit\n"
        'DATA_DIR    = PROJECT_ROOT / "data"\n'
        'RESULTS_DIR = PROJECT_ROOT / "results"\n'
        "TRAIN       = False   # True: train and save weights | False: load results/models/*.pth\n"
        "EPOCHS      = 10\n"
        "BATCH_SIZE  = 32\n"
        + ("\nutils.set_seed(42)\ndevice = utils.get_device()" if "utils" in modules else "")))
    for i, section in enumerate(sections, start=1):
        cells.append(nbf.v4.new_markdown_cell(f"## {i}. {section}\n\n*What this step does and why.*"))
        if section.lower() != "discussion":
            cells.append(nbf.v4.new_code_cell("# call src functions here (1-8 lines, no def/class)"))
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nbf.writes(nb)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--title", default="ML project")
    parser.add_argument("--modules", nargs="+", default=["data", "models", "training", "plots", "utils"])
    parser.add_argument("--sections", nargs="+",
                        default=["Data", "Models", "Training", "Evaluation", "Discussion"])
    args = parser.parse_args()
    root = args.project_dir

    listing = "\n".join(f"    {m:10s} {MODULE_DOCS.get(m, 'TODO: describe')}" for m in args.modules)
    write(root / "src" / "__init__.py", f'"""{args.title}.\n\nModules (in pipeline order):\n{listing}\n"""\n')
    for m in args.modules:
        write(root / "src" / f"{m}.py", UTILS if m == "utils" else f'"""{MODULE_DOCS.get(m, "TODO: describe this module.")}"""\n')
    write(root / "notebooks" / "main.ipynb", notebook(args.title, args.modules, args.sections))
    for sub in ["models", "figures"]:
        write(root / "results" / sub / ".gitkeep", "")
    write(root / "requirements.txt", REQUIREMENTS)
    write(root / ".gitignore", GITIGNORE)
    write(root / "README.md", f"# {args.title}\n\n## Structure\n\n## How to run\n\n## Results\n\n## Known issues\n")


if __name__ == "__main__":
    main()
