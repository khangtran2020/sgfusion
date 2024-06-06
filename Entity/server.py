"""
    Abstract class for Sever
    Server has list of models and clients.
    Server has methods to: aggregate model, broad cast model to a list of clients.
"""

import numpy as np
from typing import List
from rich.progress import Progress
from utils.console import console


class Server(object):

    def __init__(self, num_cluster: int, num_client: int) -> None:
        self.num_cluster = num_cluster
        self.num_client = num_client

    def init_model(self) -> None:
        pass

    def broadcast_params(self) -> None:
        pass

    def get_clients(self) -> List:
        pass

    def evaluate(self, split: str, progress: Progress) -> float:
        num_data = 0
        total_loss = 0
        task = progress.add_task(f"[green]Evaluating {split}...", total=self.num_client)
        for _, client in self.clients.items():
            n_data, loss = client.eval_loss(split=split)
            num_data += n_data
            total_loss += loss
            progress.update(task, advance=1)
        progress.stop_task(task)
        console.log(f"Done evaluating {split} for Server: :white_check_mark:")
        return np.sqrt(total_loss / (num_data + 1e-12))
