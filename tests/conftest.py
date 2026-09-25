"""Shared fixtures: a tiny synthetic dataset and small models (64x64 images, CPU)."""

import numpy as np
import pytest
import torch

from src import data
from tests.synthetic import make_dataset

SIZE = 64  # small images keep the tests fast; every model and method must work at any multiple of 16


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("xrays"), per_class=(12, 4, 4), seed=0)


@pytest.fixture(scope="session")
def sets(data_dir):
    return data.load_splits(data_dir, size=SIZE)


@pytest.fixture
def images():
    torch.manual_seed(0)
    return torch.rand(2, 1, SIZE, SIZE) * 2 - 1, torch.tensor([0, 1])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
