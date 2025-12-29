"""隨機種子設置模塊

提供設置隨機種子的功能
用於確保實驗的可重現性
"""
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """設置所有隨機種子

    為Python、NumPy和PyTorch設置相同的隨機種子
    以確保實驗結果的可重現性

    Args:
        seed: 隨機種子值
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
