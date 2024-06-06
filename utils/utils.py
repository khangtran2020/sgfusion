import os
import pickle
import random
import torch
import numpy as np
import pandas as pd
from typing import Dict
from utils.console import log_table


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_dict(path: str, dct: Dict):
    with open(path, "wb") as f:
        pickle.dump(dct, f)


def init_history():

    history = {
        "tr_loss": [],
        "te_loss": [],
    }

    return history


def print_args(args):
    arg_dict = {}
    for key in vars(args).keys():
        arg_dict[f"{key}"] = f"{getattr(args, key)}"
    log_table(dct=arg_dict, name="Arguments")
    return arg_dict


def get_index_by_value(a, val):
    return (a == val).nonzero(as_tuple=True)[0]
