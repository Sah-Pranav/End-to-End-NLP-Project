from __future__ import annotations

import random

import numpy as np
import torch

from summarization.logging_utils import get_logger


logger = get_logger(__name__)


def set_global_seed(
    seed: int,
    deterministic: bool = True,
) -> None:
    """Set global random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.info(
        "Global random seed set to %d "
        "(deterministic=%s).",
        seed,
        deterministic,
    )