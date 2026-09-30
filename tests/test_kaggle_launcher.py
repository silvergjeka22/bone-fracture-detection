"""The Kaggle launcher, offline: the folder it sends to Kaggle and the code packed in the notebook."""

import json
import shutil

import pytest

from scripts import kaggle_run


def test_build_packs_the_code_and_uses_your_account(tmp_path):
    kernel = kaggle_run.build(tmp_path / "kernel", quick=True, user="someone")
    assert kernel == "someone/bone-fracture-detection"
    meta = json.loads((tmp_path / "kernel" / "kernel-metadata.json").read_text())
    assert meta["id"] == kernel and meta["is_private"] and meta["enable_gpu"] and meta["code_file"] == "main.ipynb"
    assert "mahmudulhasantasin/fracatlas-original-dataset" in meta["dataset_sources"]

    notebook = json.loads((tmp_path / "kernel" / "main.ipynb").read_text())
    unpack = notebook["cells"][0]["source"]
    assert "QUICK      = True" in "".join("".join(c["source"]) for c in notebook["cells"])
    assert all(c.get("outputs", []) == [] for c in notebook["cells"])

    # the first cell really unpacks src/ (here into tmp_path instead of /tmp/bfd)
    exec(unpack.replace(kaggle_run.UNPACK, (tmp_path / "code").as_posix()), {})
    for name in ("src/data.py", "src/detect.py", "src/bootstrap.py", "requirements.txt"):
        assert (tmp_path / "code" / name).read_text() == (kaggle_run.ROOT / name).read_text()


def test_a_token_in_the_code_stops_the_upload(tmp_path):
    root = tmp_path / "project"
    for name in ("src", "notebooks"):
        shutil.copytree(kaggle_run.ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("requirements.txt", "kernel-metadata.json"):
        shutil.copy(kaggle_run.ROOT / name, root / name)
    (root / "src" / "leak.py").write_text('TOKEN = "ghp_' + "a" * 30 + '"\n')
    with pytest.raises(SystemExit):
        kaggle_run.build(tmp_path / "kernel", quick=False, user="someone", root=root)


def test_username_from_the_flag_or_the_environment(monkeypatch):
    assert kaggle_run.username("given") == "given"
    monkeypatch.setenv("KAGGLE_USERNAME", "from-env")
    assert kaggle_run.username() == "from-env"
