"""
    Abstract class for client
    Each client will have a set of data
    Each client will have a model
    Each client will have methods for: updating model, receive model params, send model params
"""

import torch
import pickle
from utils.console import console
from Models.model import LSTMTarget
from collections import OrderedDict
from typing import Tuple


class Client(object):

    def __init__(
        self,
        cid: int,
        data_path: str,
        updating_steps: int,
        device: torch.device,
    ) -> None:
        self.data_path = data_path
        self.cid = cid
        self.updating_steps = updating_steps
        self.device = device
        self.cluster = -1

    def set_data(self) -> None:
        with console.status(f"Initializing Data for Client {self.cid}") as status:
            with open(f"{self.data_path}_train.pkl", "rb") as pkl:
                self.tr_loader = pickle.load(pkl)
            with open(f"{self.data_path}_val.pkl", "rb") as pkl:
                self.te_loader = pickle.load(pkl)
            console.log(f"Done Reading data for client {self.cid}: :white_check_mark:")

    def init_model(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        hidden_dim: int,
        context_dim: int,
        lr: float,
    ) -> None:
        with console.status(f"Initializing Model for Client {self.cid}") as status:
            self.model = LSTMTarget(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
                hidden_dim=hidden_dim,
                context_final_dim=context_dim,
            )
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=lr, weight_decay=1e-4
            )
            console.log(f"Done Init model for client {self.cid}: :white_check_mark:")

    def train(self) -> None:
        pass

    def set_params(self, params: OrderedDict) -> None:
        self.model.load_state_dict(params)

    def get_params(self) -> OrderedDict:
        return self.model.state_dict()

    def eval_loss(self, split: str) -> Tuple[int, float]:
        with torch.no_grad():
            loader = self.tr_loader if split == "train" else self.te_loader
            num_data = 0
            total_loss = 0
            for batch in loader:
                inputs, target = self.model.embed_inputs(batch, device=self.device)
                inputs = inputs.float().to(self.device)
                out = self.model.forward(inputs, device=self.device)
                loss = (out - target.to(self.device)).pow(2).sum()
                total_loss += loss.item()
                num_data += target.size(dim=0)
            console.log(
                f"Done evaluating {split} cluster for client {self.cid}: :white_check_mark:"
            )
        return num_data, total_loss
