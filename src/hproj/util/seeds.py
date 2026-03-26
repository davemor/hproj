import hashlib
import os
import random

import cupy as cp
import numpy as np
import torch


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    cp.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


# (base_seed, dataset, encoder, projector, dimension, fold) -> seed
# umap.UMAP(random_state=seed)
def make_seed(base_seed, *keys):
    key = "|".join(map(str, keys))
    h = hashlib.blake2b(key.encode(), digest_size=4).digest()
    return (int.from_bytes(h, "little") + base_seed) % (2**32)
