import random

import numpy as np
import torch

from summarization.reproducibility import set_global_seed


def test_set_global_seed_produces_reproducible_values():
    seed = 42

    set_global_seed(seed)

    first_python = random.random()
    first_numpy = np.random.rand()
    first_torch = torch.rand(1).item()

    set_global_seed(seed)

    second_python = random.random()
    second_numpy = np.random.rand()
    second_torch = torch.rand(1).item()

    assert first_python == second_python
    assert first_numpy == second_numpy
    assert first_torch == second_torch