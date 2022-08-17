import numpy as np
import random as rd
import torch
import sys
import os

sys.path.append(os.getcwd())

from slot_attention.params import SlotAttentionParams
from slot_attention.train import main


if __name__ == "__main__":
    for seed in range(2022, 2027):
        np.random.seed(seed)
        rd.seed(seed)
        torch.manual_seed(seed)
        params = SlotAttentionParams(max_epochs=150)
        main(f"commnist_tsla_{seed}", params)
