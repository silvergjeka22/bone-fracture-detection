"""Run notebooks/bone-fracture-detection.ipynb on a Kaggle GPU and download the results (macOS, Linux and Windows).

    push [--quick] [--user NAME]   pack src/ into the notebook and start it on YOUR Kaggle account
    status                         is it running, finished or failed?
    get                            download the output into out/run-<time>/ and copy results/ into ./results

The code travels inside the notebook (no GitHub token, no extra Kaggle dataset). The notebook is
<your-kaggle-username>/bone-fracture-detection; the dataset and the GPU come from kernel-metadata.json.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "bone-fracture-detection"
BUILD = ROOT / ".kaggle-build"      # the folder sent to Kaggle (kept for inspection)
LAST = ROOT / ".kaggle-kernel"      # the notebook last pushed from this computer, for status / get
UNPACK = "/tmp/bfd"                 # where the notebook unpacks the code (the setup cell looks there)
SECRET = re.compile(r"gh[pousr]_\w{20,}|github_pat_\w{20,}|KGAT_[\w-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def kaggle(*args, quiet=False, strict=True):
    """Run the Kaggle command-line tool and return its output; stop with a hint if it fails.

    strict=False: the word "error" in the output is not a failure (status / output of a failed run).
    """
    exe = shutil.which("kaggle")
    cmd = [exe] if exe else [sys.executable, "-c", "import sys; from kaggle.cli import main; sys.argv[0] = 'kaggle'; main()"]
    try:
        done = subprocess.run(cmd + list(args), capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as error:
        sys.exit(f"Cannot start the Kaggle tool ({error}). Install it: python -m pip install kaggle")
    out = (done.stdout + done.stderr).strip()
    if out and not quiet:
        print(out)
    if "No module named 'kaggle'" in out:
        sys.exit("The Kaggle tool is not installed. Run: python -m pip install kaggle")
    if (done.returncode != 0 or re.search(r"401|403|Unauthorized|Forbidden|Could not find kaggle", out, re.IGNORECASE)
            or (strict and re.search(r"\berror\b", out, re.IGNORECASE))):
        sys.exit(help_for(out))
    return out


def help_for(out):
    """A hint for the usual Kaggle errors."""
    if re.search(r"401|Unauthorized|credentials|kaggle\.json|authenticate", out, re.IGNORECASE):
        return ("Kaggle login failed. Create a token on kaggle.com > Settings > API > Create New Token and put "
                "kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<you>\\.kaggle\\).")
    if re.search(r"403|Forbidden|permission", out, re.IGNORECASE):
        return ("Kaggle refused the request: check that --user is YOUR username and that your account is "
                "phone-verified (needed for GPU and internet).")
    return "The Kaggle tool reported an error (see above)."


def username(given=None):
    """Kaggle username: --user, then KAGGLE_USERNAME, then the logged-in account, then the last push."""
    if given or os.environ.get("KAGGLE_USERNAME"):
        return given or os.environ["KAGGLE_USERNAME"]
    try:
        found = re.search(r"username:\s*(\S+)", kaggle("config", "view", quiet=True))
    except SystemExit:
        found = None
    if found and found.group(1) != "None":
        return found.group(1)
    if LAST.exists():
        return LAST.read_text().split("/")[0]
    sys.exit("Kaggle username not found. Run: ./run.sh push --user <your-kaggle-username>")


def build(stage, quick, user, root=ROOT):
    """Write the folder for Kaggle: the notebook (code packed in its first cell) + kernel-metadata.json."""
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((root / "src").glob("*.py")) + [root / "requirements.txt"]:
            text = path.read_text(encoding="utf-8")
            if SECRET.search(text):
                sys.exit(f"{path.name} looks like it contains a password or token: nothing was uploaded.")
            bundle.writestr(path.relative_to(root).as_posix(), text)
    code = base64.b64encode(packed.getvalue()).decode()

    notebook = json.loads((root / "notebooks" / f"{SLUG}.ipynb").read_text(encoding="utf-8"))
    switched = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source, n = re.subn(r"^QUICK\s*=\s*(True|False)", f"QUICK      = {quick}", "".join(cell["source"]),
                                flags=re.MULTILINE)
            cell.update(source=source, outputs=[], execution_count=None)
            switched += n
    if switched != 1:
        sys.exit("The Settings cell must contain exactly one line 'QUICK = ...'.")
    notebook["cells"].insert(0, {
        "cell_type": "code", "id": "unpack-code", "metadata": {}, "outputs": [], "execution_count": None,
        "source": ("# Added by ./run.sh push: the project's code (src/), unpacked where the setup cell looks for it\n"
                   "import base64, io, zipfile\n"
                   f'zipfile.ZipFile(io.BytesIO(base64.b64decode("{code}"))).extractall("{UNPACK}")')})

    metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.update(id=f"{user}/{SLUG}", title=SLUG, code_file=f"{SLUG}.ipynb")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    (stage / f"{SLUG}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata["id"]


def push(quick, user, dry_run):
    kernel = build(BUILD, quick, user or "your-username" if dry_run else username(user))
    if dry_run:
        print(f"Dry run: {kernel} prepared in {BUILD} (nothing was uploaded).")
        return
    print(f"Starting {'the quick check' if quick else 'the full run'} on https://www.kaggle.com/code/{kernel}")
    kaggle("kernels", "push", "-p", str(BUILD))
    LAST.write_text(kernel)
    print("Kaggle accepted the job: the computer can be switched off. Check it with: ./run.sh status")


def kernel_id(user):
    if user:
        return f"{user}/{SLUG}"
    return LAST.read_text().strip() if LAST.exists() else f"{username()}/{SLUG}"


def status(user):
    """Print the state of the run; True when it is complete."""
    state = kaggle("kernels", "status", kernel_id(user), strict=False).lower()
    if "error" in state:
        print("The run failed: ./run.sh get downloads its log (the .log file in out/run-.../).")
    return "complete" in state


def get(user):
    kernel = kernel_id(user)
    finished = status(user)
    out = ROOT / "out" / f"run-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    kaggle("kernels", "output", kernel, "-p", str(out), strict=False)
    if finished and (out / "results").is_dir():
        shutil.copytree(out / "results", ROOT / "results", dirs_exist_ok=True)
        print(f"Downloaded into {out} and copied results/ into {ROOT / 'results'}.")
    else:
        print(f"Downloaded into {out}. The run is not complete, so ./results was left untouched.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the notebook on Kaggle and get the results.")
    parser.add_argument("command", choices=["push", "status", "get"])
    parser.add_argument("--quick", action="store_true", help="5-minute health check on a small subset")
    parser.add_argument("--user", help="your Kaggle username (found automatically when possible)")
    parser.add_argument("--dry-run", action="store_true", help="prepare .kaggle-build/ without uploading")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # Windows consoles cannot print every character
    if args.command == "push":
        push(args.quick, args.user, args.dry_run)
    elif args.command == "status":
        status(args.user)
    else:
        get(args.user)


if __name__ == "__main__":
    main()
