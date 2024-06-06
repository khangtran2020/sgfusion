import torch
import numpy as np
from Entity.client import Client
from collections import OrderedDict
from typing import List
from utils.console import console
from rich.progress import Progress
from Models.utils import clip_grad_norm_


class ClientMCFL(Client):

    def __init__(
        self,
        cid: int,
        data_path: str,
        updating_steps: int,
        device: torch.device,
    ) -> None:
        super(ClientMCFL, self).__init__(
            cid=cid,
            data_path=data_path,
            updating_steps=updating_steps,
            device=device,
        )

    def compute_cluster(self, params_list: List[OrderedDict]) -> int:
        with console.status(f"Computing cluster for Client {self.cid}") as status:
            with torch.no_grad():
                local_state_dict = self.model.state_dict()
                d_min = np.inf
                new_cluster = -1
                for i, params in enumerate(params_list):
                    d = 0
                    for key in local_state_dict.keys():
                        d += (local_state_dict[key] - params[key]).norm(p=2).item() ** 2
                    d = np.sqrt(d)
                    if d_min > d:
                        d_min = d
                        new_cluster = i
            cluster_response = np.zeros(len(params_list))
            cluster_response[new_cluster] = 1
            console.log(
                f"Done computing cluster for client {self.cid}: :white_check_mark:"
            )
        return cluster_response

    def train(self, progress: Progress) -> None:
        self.model = self.model.to(self.device)
        self.model.train()

        task = progress.add_task("[blue]Training...", total=self.updating_steps)
        for step in range(self.updating_steps):
            for batch in self.loader:
                inputs, target = self.model.embed_inputs(batch)
                inputs = inputs.float().to(self.device)
                out = self.model(inputs)
                loss = (out - target.to(self.device)).pow(2).mean().sqrt()
                loss.backward()
                clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=0.1, norm_type=2.0
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            progress.update(task, advance=1)
        progress.stop_task(task)
        console.log(f"Done updating model for client {self.cid}: :white_check_mark:")
