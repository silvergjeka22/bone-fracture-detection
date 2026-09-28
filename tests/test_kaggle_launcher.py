"""Offline checks for the private-code Kaggle submission bundle."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts import kaggle_run


class KaggleLauncherTest(unittest.TestCase):
    def test_prepare_private_code_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            self.check_snapshot(Path(folder))

    def check_snapshot(self, stage):
        dataset_id, provenance = kaggle_run.prepare(stage, quick=True)

        kernel_config = json.loads((kaggle_run.ROOT / "kernel-metadata.json").read_text())
        owner, kernel_slug = kernel_config["id"].split("/", 1)
        expected_prefix = f"{owner}/{kernel_slug[:20]}-code-"
        self.assertTrue(dataset_id.startswith(expected_prefix))
        self.assertIs(provenance["quick"], True)
        self.assertIn("src/bootstrap.py", provenance["files"])
        self.assertTrue(all(".git" not in path for path in provenance["files"]))

        dataset_meta = json.loads((stage / "dataset" / "dataset-metadata.json").read_text())
        kernel_meta = json.loads((stage / "kernel" / "kernel-metadata.json").read_text())
        notebook = json.loads((stage / "kernel" / "main.ipynb").read_text())
        notebook_source = "\n".join(
            "".join(cell.get("source", "")) for cell in notebook["cells"]
        )

        self.assertEqual(dataset_meta["id"], dataset_id)
        self.assertIn(dataset_id, kernel_meta["dataset_sources"])
        self.assertIs(kernel_meta["is_private"], True)
        self.assertNotIn("GITHUB_TOKEN", notebook_source)
        self.assertIn(
            f'BFD_CODE_SHA256 = "{provenance["snapshot_sha256"]}"', notebook_source
        )
        self.assertIn("QUICK      = True", notebook_source)

    def test_remote_snapshot_must_have_the_exact_checksum_filename(self):
        digest = "a" * 64
        expected = f"bfd-code-{digest}.json"
        api = SimpleNamespace(
            dataset_list_files=lambda *_args, **_kwargs: SimpleNamespace(
                dataset_files=[SimpleNamespace(name=expected)]
            )
        )
        kaggle_run.require_snapshot_file(api, "owner/code-aaaaaaaaaaaaaaaa", digest)

        api.dataset_list_files = lambda *_args, **_kwargs: SimpleNamespace(
            dataset_files=[SimpleNamespace(name="wrong.json")]
        )
        with self.assertRaisesRegex(RuntimeError, "no notebook was submitted"):
            kaggle_run.require_snapshot_file(api, "owner/code-aaaaaaaaaaaaaaaa", digest)
