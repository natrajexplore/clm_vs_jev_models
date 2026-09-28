import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs. Log the seed with every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
