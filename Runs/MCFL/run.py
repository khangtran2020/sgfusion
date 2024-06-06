import torch
from utils.console import console
from argparse import Namespace
from typing import Dict
from rich.progress import Progress
from Runs.MCFL.Entity.server import ServerMCFL
from Runs.MCFL.Entity.client import ClientMCFL


def run(args: Namespace, data_dict: Dict, device: torch.device, history: Dict):

    # Create server
    server = ServerMCFL(num_cluster=args.num_cluster, num_client=args.num_client)
    server.init_model(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
    )

    clients = []
    for i in range(args.num_client):
        client = ClientMCFL(
            cid=data_dict[i]["cid"],
            data_path=data_dict[i]["path"],
            batch_size=args.batch_size,
            updating_steps=args.client_updating_step,
            device=device,
        )
        client.set_data()
        client.init_model(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            context_dim=args.context_dim,
            lr=args.lr,
        )
        clients.append(client)

    server.init_client(clients=clients)
    with Progress(console=console) as progress:
        task = progress.add_task(
            "[yellow]Global training...", total=args.num_global_step
        )
        for step in range(args.num_global_step):
            server.compute_centroid(progress=progress)
            server.broadcast_params(progress=progress)
            for client in server.clients:
                client.train(progress=progress)
            if (step % args.eval_step) == 0:
                train_loss = server.evaluate(split="train", progress=progress)
                test_loss = server.evaluate(split="test", progress=progress)
                console.log(
                    "Step {}: train loss - {} | test loss - {}".format(
                        step, train_loss, test_loss
                    )
                )
                history["tr_loss"].append(train_loss)
                history["te_loss"].append(test_loss)
            progress.update(task, advance=1)