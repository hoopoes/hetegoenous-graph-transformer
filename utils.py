# utility functions for metapath2vec
from typing import List, Dict, Tuple, Set, Union, Optional, Callable, Iterable

import numpy as np
import pandas as pd
from configparser import ConfigParser

import torch


def get_memory_stat(device):
    # reserved memory: allocated memory + pre-cached memory
    # allocated memory: memory that is actually used by pytorch
    total_memory = np.round(torch.cuda.get_device_properties(device=device).total_memory / 1024**2)
    reserved_memory = np.round(torch.cuda.memory_reserved(device=device) / 1024**2)
    allocated_memory = np.round(torch.cuda.memory_allocated(device=device) / 1024**2)

    report = pd.DataFrame(
        {'category': ['total', 'reserved', 'allocated'],
         'memory_stat': [total_memory, reserved_memory, allocated_memory]}
    )
    return report


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_num_params(model):
    return sum([p.numel() for p in model.parameters()])
