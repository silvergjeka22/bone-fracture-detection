"""Launch this project on Kaggle without sending GitHub credentials.

The launcher creates an immutable private Kaggle Dataset containing an
allowlisted snapshot of the local Python sources.  The notebook receives that
dataset as an input, reconstructs the package in /tmp, and writes its durable
results to /kaggle/working/results.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERN = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"KGAT_[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def kernel_metadata() -> dict:
    metadata = json.loads((ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+", metadata.get("id", "")):
        raise ValueError("Set kernel-metadata.json id to <kaggle-user>/<notebook-slug>.")
    if metadata.get("is_private") is not True:
        raise ValueError("kernel-metadata.json must contain is_private: true.")
    metadata.pop("id_no", None)  # Kaggle otherwise gives the numeric ID precedence over the slug.
    return metadata


def source_files() -> dict[str, str]:
    """Return the deliberate source allowlist; credentials and Git metadata never enter it."""
    src = ROOT / "src"
    if src.is_symlink() or not src.is_dir():
        raise ValueError("src must be a real local directory.")
    paths = sorted(src.rglob("*.py")) + [ROOT / "requirements.txt"]
    files: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != ROOT):
            raise ValueError(f"Symlinks cannot be packaged: {relative}")
        path.resolve().relative_to(ROOT.resolve())
        if re.search(r"token|secret|credential|\.env", path.name, re.IGNORECASE):
            raise ValueError(f"Credential-like filename selected for upload: {relative}")
        text = path.read_text(encoding="utf-8")
        if SECRET_PATTERN.search(text):
            raise ValueError(f"Possible embedded credential in {relative}; upload cancelled.")
        if path.suffix == ".py":
            ast.parse(text, filename=str(relative))
        files[relative.as_posix()] = text
    if "src/__init__.py" not in files or "src/bootstrap.py" not in files:
        raise ValueError("The source snapshot is incomplete.")
    return files


def prepare(stage: Path, quick: bool) -> tuple[str, dict]:
    """Prepare a code Dataset plus kernel in stage without network access."""
    metadata = kernel_metadata()
    branch = git_value("branch", "--show-current")
    requested_branch = os.environ.get("BRANCH")
    if requested_branch and requested_branch != branch:
        raise ValueError(
            f"BRANCH={requested_branch} differs from the checked-out branch {branch}. "
            "Switch branches locally first; the launcher uploads local files."
        )

    files = source_files()
    payload = json.dumps(
        {"format": 1, "files": files}, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    owner, notebook_slug = metadata["id"].split("/", 1)
    dataset_slug = f"{notebook_slug[:20]}-code-{digest[:16]}"
    dataset_id = f"{owner}/{dataset_slug}"

    dataset_dir = stage / "dataset"
    kernel_dir = stage / "kernel"
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)
    (dataset_dir / f"bfd-code-{digest}.json").write_bytes(payload)
    write_json(
        dataset_dir / "dataset-metadata.json",
        {
            "id": dataset_id,
            "title": dataset_slug,
            "licenses": [{"name": "unknown"}],
            "description": (
                "Private source snapshot for a Kaggle job. "
                f"SHA256: {digest}. The launcher excludes credentials and Git metadata."
            ),
        },
    )

    notebook = json.loads((ROOT / "notebooks/main.ipynb").read_text(encoding="utf-8"))
    code_markers = quick_markers = 0
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", ""))
        if cell.get("cell_type") == "code":
            source, count = re.subn(
                r'^BFD_CODE_SHA256 = ""$', f'BFD_CODE_SHA256 = "{digest}"', source, flags=re.MULTILINE
            )
            code_markers += count
            source, count = re.subn(
                r"^QUICK\s*=\s*(?:True|False)\b", f"QUICK      = {quick}", source, flags=re.MULTILINE
            )
            quick_markers += count
            ast.parse(source)
            cell["outputs"] = []
            cell["execution_count"] = None
        if SECRET_PATTERN.search(source):
            raise ValueError("Possible embedded credential in notebook source; upload cancelled.")
        cell["source"] = source.splitlines(keepends=True)
        cell["metadata"] = {}
        cell.pop("attachments", None)
    if (code_markers, quick_markers) != (1, 1):
        raise ValueError("Expected one BFD_CODE_SHA256 marker and one QUICK setting in the notebook.")

    provenance = {
        "branch": branch,
        "commit": git_value("rev-parse", "HEAD"),
        "snapshot_sha256": digest,
        "dataset": dataset_id,
        "quick": quick,
        "files": list(files),
    }
    notebook["metadata"] = {
        key: value
        for key, value in notebook.get("metadata", {}).items()
        if key in ("kernelspec", "language_info")
    }
    notebook["metadata"]["bfd_submission"] = provenance
    metadata["code_file"] = "main.ipynb"
    metadata["dataset_sources"] = list(
        dict.fromkeys([*metadata.get("dataset_sources", []), dataset_id])
    )
    write_json(kernel_dir / "main.ipynb", notebook)
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    write_json(stage / "manifest.json", provenance)
    return dataset_id, provenance


def kaggle_api():
    try:
        from kaggle import api
    except ImportError as exc:
        raise RuntimeError("Install Kaggle CLI: python -m pip install 'kaggle>=2.2.4,<3'") from exc
    api.authenticate()
    return api


def dataset_info(api, dataset_id: str):
    """Return the dataset, accounting for Kaggle hiding absent private slugs."""
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    from requests import HTTPError

    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = dataset_id.split("/", 1)
    try:
        with api.build_kaggle_client() as client:
            return client.datasets.dataset_api_client.get_dataset(request)
    except HTTPError as exc:
        # Kaggle can answer 403, rather than 404, when an authenticated owner
        # asks for a private dataset slug that has not been created yet.  The
        # create call may proceed, but submit() still re-reads the dataset and
        # verifies is_private=True before it submits any notebook.
        if exc.response is not None and exc.response.status_code in (403, 404):
            return None
        raise


def require_private(info) -> None:
    if info is None or info.is_private is not True:
        raise RuntimeError("The code dataset is not private; no notebook was submitted.")


def require_snapshot_file(api, dataset_id: str, digest: str) -> None:
    """Reject a reused slug if its remote contents do not match our immutable layout."""
    expected = f"bfd-code-{digest}.json"
    response = api.dataset_list_files(dataset_id, page_size=20)
    names = sorted(item.name for item in (response.dataset_files or []))
    if names != [expected]:
        raise RuntimeError(
            f"Code dataset contents are {names!r}, expected {[expected]!r}; "
            "no notebook was submitted."
        )


def require_success(response, operation: str) -> None:
    if response is None or getattr(response, "error", None):
        detail = getattr(response, "error", None) or "empty API response"
        raise RuntimeError(f"{operation} failed: {detail}")


def submit(api, stage: Path, dataset_id: str, digest: str, wait_seconds: int) -> None:
    info = dataset_info(api, dataset_id)
    if info is None:
        print("Uploading an immutable private code snapshot...", flush=True)
        response = api.dataset_create_new(
            str(stage / "dataset"), public=False, quiet=False, convert_to_csv=False
        )
        require_success(response, "Code upload")
    else:
        require_private(info)
        print("Reusing the identical private code snapshot.", flush=True)

    deadline = time.monotonic() + wait_seconds
    while True:
        status = api.dataset_status(dataset_id)
        if status == "ready":
            break
        if status not in ("pending", "creating"):
            raise RuntimeError(f"Code dataset status is {status!r}; no notebook was submitted.")
        if time.monotonic() >= deadline:
            raise RuntimeError("Code dataset processing timed out; no notebook was submitted.")
        print(f"Waiting for code dataset: {status}", flush=True)
        time.sleep(min(5, max(0, deadline - time.monotonic())))

    require_private(dataset_info(api, dataset_id))
    require_snapshot_file(api, dataset_id, digest)
    response = api.kernels_push(str(stage / "kernel"))
    require_success(response, "Notebook submission")
    for field in (
        "invalid_dataset_sources",
        "invalid_competition_sources",
        "invalid_kernel_sources",
        "invalid_model_sources",
    ):
        if getattr(response, field, None):
            raise RuntimeError(f"Kaggle rejected an attached input: {field}")
    print(f"Submitted Kaggle version {response.version_number}: {response.url}")
    print("Kaggle accepted the job; the PC can now be switched off.")


def get_outputs(api, kernel: str) -> None:
    """Download into a new folder, never erasing a previous result."""
    output_root = ROOT / "out"
    output_root.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(tempfile.mkdtemp(prefix=f"run-{stamp}-", dir=output_root))
    api.kernels_output(kernel, str(destination), quiet=False)
    status = str(api.kernels_status(kernel).status).rsplit(".", 1)[-1].lower()
    print(f"Downloaded the latest available output to {destination}")
    if status == "complete" and (destination / "results").is_dir():
        shutil.copytree(destination / "results", ROOT / "results", dirs_exist_ok=True)
        print("Copied completed results into ./results.")
    else:
        print(f"Kernel status: {status}. Existing ./results was left untouched.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local project privately on Kaggle without a GitHub token."
    )
    commands = parser.add_subparsers(dest="command")
    push = commands.add_parser("push", help="upload local code privately and start a remote job")
    push.add_argument("--quick", action="store_true", help="run the small health check")
    push.add_argument(
        "--dry-run", action="store_true", help="prepare a local preview without Kaggle/network"
    )
    push.add_argument("--wait-seconds", type=int, default=300)
    commands.add_parser("status", help="show the latest job status")
    commands.add_parser("get", help="download outputs without deleting earlier downloads")
    commands.add_parser("slides", help="rebuild the presentation")
    commands.add_parser("stop", help="print the Kaggle page used to stop a session")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "slides":
            subprocess.run(["make", "-C", str(ROOT / "presentation")], check=True)
        elif args.command == "push":
            if args.wait_seconds <= 0:
                raise ValueError("--wait-seconds must be positive.")
            builds = ROOT / ".kaggle-build"
            builds.mkdir(exist_ok=True)
            if args.dry_run:
                stage = Path(tempfile.mkdtemp(prefix="preview-", dir=builds))
                dataset_id, provenance = prepare(stage, args.quick)
                print(f"Prepared {dataset_id}")
                print("Files: " + ", ".join(provenance["files"]))
                print(f"Dry run only; inspect {stage}")
            else:
                with tempfile.TemporaryDirectory(prefix="upload-", dir=builds) as folder:
                    stage = Path(folder)
                    dataset_id, provenance = prepare(stage, args.quick)
                    print(f"Local branch: {provenance['branch'] or '(detached)'}")
                    print(f"Private code dataset: {dataset_id}")
                    submit(
                        kaggle_api(),
                        stage,
                        dataset_id,
                        provenance["snapshot_sha256"],
                        args.wait_seconds,
                    )
        else:
            kernel = kernel_metadata()["id"]
            if args.command == "stop":
                print(f"Open https://www.kaggle.com/code/{kernel} and click Stop Session.")
            elif args.command == "status":
                print(f"{kernel}: {kaggle_api().kernels_status(kernel).status}")
            elif args.command == "get":
                get_outputs(kaggle_api(), kernel)
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 403:
            print(
                "Error: Kaggle denied this operation (403). Move any legacy "
                "~/.kaggle/kaggle.json aside, run 'kaggle auth login --force', "
                "and accept Dataset/Notebook permissions. No notebook was submitted.",
                file=sys.stderr,
            )
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
