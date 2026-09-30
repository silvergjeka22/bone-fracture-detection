"""Shared fixtures: a tiny synthetic FracAtlas and its split (64x64 images, CPU)."""

import numpy as np
import pytest
import torch

from src import data
from tests.synthetic import make_dataset

SIZE = 64  # small images keep the tests fast; every model and method must work at any multiple of 16


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("fracatlas"), per_class=16, seed=0)


@pytest.fixture(scope="session")
def dataset(data_dir):
    return data.load_dataset(data_dir, size=SIZE)


@pytest.fixture(scope="session")
def dup(dataset):
    return data.study_duplicates(dataset)


@pytest.fixture(scope="session")
def split(dataset, dup):
    return data.split_indices(dataset.labels, dup["groups"], seed=0)


@pytest.fixture(scope="session")
def sets(dataset, split):
    return {name: dataset.subset(idx) for name, idx in split.items()}


@pytest.fixture
def images():
    torch.manual_seed(0)
    return torch.rand(2, 1, SIZE, SIZE) * 2 - 1, torch.tensor([0, 1])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
