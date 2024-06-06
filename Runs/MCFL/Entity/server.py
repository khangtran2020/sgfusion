import os
import gc
import torch
import numpy as np
from Entity.server import Server
from Models.model import LSTMTarget
from Runs.MCFL.Entity.client import ClientMCFL
from typing import List
from collections import OrderedDict
from utils.console import console
from rich.progress import Progress
from utils.utils import save_dict
from copy import deepcopy


class ServerMCFL(Server):

    def __init__(self, num_cluster: int, num_client: int, name: str) -> None:
        super(ServerMCFL, self).__init__(num_client=num_client, num_cluster=num_cluster)
        self.client_dict = {}
        self.name = name

    def init_model(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        hidden_dim: int,
        context_dim: int,
    ) -> None:
        with console.status(f"Initialize model for Server") as status:
            self.models = {}
            for i in range(self.num_cluster):
                self.models[f"Cluster_{i}"] = LSTMTarget(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    dropout=dropout,
                    hidden_dim=hidden_dim,
                    context_final_dim=context_dim,
                )
            console.log(f"Done init model for Server: :white_check_mark:")

    def init_client(self, clients: List[ClientMCFL]):
        with console.status(f"Initialize model for Server") as status:
            self.clients = {}
            for client in clients:
                self.clients[f"Client_{client.cid}"] = client
            console.log(f"Done init clients for Server: :white_check_mark:")

    def get_model_params(self) -> List[OrderedDict]:
        params = []
        for i in range(self.num_cluster):
            params.append(self.models[f"Cluster_{i}"].state_dict())
        return params

    def compute_centroid(self, progress: Progress):
        with torch.no_grad():
            params = self.get_model_params()
            clustered_client = {}

            for i in range(self.num_cluster):
                clustered_client[f"Cluster_{i}"] = []

            task = progress.add_task(
                "[green]Client responding...", total=self.num_client
            )
            for i in range(self.num_client):
                cluster_response = self.clients[f"Client_{i}"].compute_cluster(
                    params_list=params
                )
                idx = np.argmax(cluster_response)
                clustered_client[f"Cluster_{idx}"].append(self.clients[f"Client_{i}"])
                progress.update(task, advance=1)
            progress.stop_task(task)
            progress.update(task, visible=False)

            task = progress.add_task(
                "[green]Computing cluster model...", total=self.num_client
            )
            for i in range(self.num_cluster):
                curr_cluster_model = None
                for j, client in enumerate(clustered_client[f"Cluster_{i}"]):
                    param = client.get_params()
                    if j == 0:
                        curr_cluster_model = deepcopy(param)
                    else:
                        for key in param.keys():
                            curr_cluster_model[key] = (
                                curr_cluster_model[key] + param[key]
                            )
                for key in param.keys():
                    curr_cluster_model[key] = curr_cluster_model[key] / len(
                        clustered_client[f"Cluster_{i}"]
                    )
                self.models[f"Cluster_{i}"] = curr_cluster_model
                del curr_cluster_model
                gc.collect()
                progress.update(task, advance=1)
            progress.stop_task(task)
            progress.update(task, visible=False)

            console.log(f"Done compute cluster models for Server: :white_check_mark:")
            self.client_dict = clustered_client

    def broadcast_params(self, progress: Progress) -> None:
        with torch.no_grad():
            task = progress.add_task("[red]Broadcasting...", total=self.num_cluster)
            for i in range(self.num_cluster):
                params = self.models[f"Cluster_{i}"].state_dict()
                for client in self.client_dict["Cluster_{i}"]:
                    client.set_params(params=params)
                progress.update(task, advance=1)
            progress.stop_task(task)
            progress.update(task, visible=False)
            console.log(
                f"Done broadcasting cluster models for Server: :white_check_mark:"
            )

    def save_cluster(self):
        save_dict(
            os.path.join("results/models", f"{self.name}-cluster.pkl"),
            dct=self.client_dict,
        )
        for key, model in self.models.items():
            model = model.to("cpu")
            torch.save(model, os.path.join("results/model", f"{self.name}-{key}.pt"))
