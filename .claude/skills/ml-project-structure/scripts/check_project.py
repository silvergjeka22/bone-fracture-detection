"""Check that a project follows the "code in src/, thin notebook" structure.

    python check_project.py <project_dir> [--notebook notebooks/main.ipynb] [--max-lines 10]

Errors (exit code 1): functions/classes defined in the notebook, absolute local paths,
secret-looking strings, the same function name defined in two src modules.
Warnings: long code cells (the first two cells, setup and settings, are exempt), big outputs.
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

import nbformat

SECRET = re.compile(r"""(api[_-]?key|token|secret|password|"key")\s*[:=]\s*['"]?[A-Za-z0-9_\-]{16,}""", re.I)
ABS_PATH = re.compile(r"""['"](?:[A-Za-z]:\\\\|[A-Za-z]:\\|/Users/|/home/)""")


def check_notebook(path, max_lines):
    errors, warnings = [], []
    nb = nbformat.read(path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    for n, cell in enumerate(code_cells, start=1):
        src = cell.source
        live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("#", "%", "!")))
        if re.search(r"^\s*(def|class)\s+\w+", live, re.M):
            errors.append(f"code cell {n}: defines a function/class, move it to src/")
        if ABS_PATH.search(src):
            errors.append(f"code cell {n}: absolute local path, derive it from PROJECT_ROOT instead")
        if SECRET.search(src):
            errors.append(f"code cell {n}: looks like a secret (API key/token), remove and revoke it")
        lines = len([l for l in src.splitlines() if l.strip()])
        if n > 2 and lines > max_lines:
            warnings.append(f"code cell {n}: {lines} lines (> {max_lines}), move logic into a src function")
    size_mb = path.stat().st_size / 1e6
    if size_mb > 5:
        warnings.append(f"{path.name} is {size_mb:.0f} MB, clear outputs and save figures to results/figures/")
    return errors, warnings


def check_src(src_dir):
    defined = defaultdict(list)
    for file in sorted(src_dir.glob("*.py")):
        for node in ast.parse(file.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                defined[node.name].append(file.name)
    return [f"'{name}' is defined in {', '.join(files)}: keep one implementation"
            for name, files in defined.items() if len(files) > 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--notebook", default="notebooks/main.ipynb")
    parser.add_argument("--max-lines", type=int, default=10)
    args = parser.parse_args()

    notebook = args.project_dir / args.notebook
    errors, warnings = check_notebook(notebook, args.max_lines) if notebook.exists() else ([f"{notebook} not found"], [])
    if (args.project_dir / "src").is_dir():
        errors += check_src(args.project_dir / "src")
    else:
        errors.append("src/ not found")

    for w in warnings:
        print(f"WARNING  {w}")
    for e in errors:
        print(f"ERROR    {e}")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
