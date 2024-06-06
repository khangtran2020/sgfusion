import argparse


def add_general_group(group):
    group.add_argument("--pname", type=str, default="", help="", required=True)
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--mode", type=str, default="")


def add_data_group(group):
    group.add_argument("--data_path", type=str, default="")
    group.add_argument("--country", type=str, default="")
    group.add_argument("--num_cluster", type=int, default=16)


def add_model_group(group):
    group.add_argument("--model", type=str, default="NN", help="Model type")
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument("--hidden_dim", type=int, default=64)
    group.add_argument("--context_dim", type=int, default=32)
    group.add_argument("--num_global_step", type=int, default=100, help="training step")
    group.add_argument(
        "--client_updating_step", type=int, default=5, help="training step of client"
    )
    group.add_argument("--eval_step", type=int, default=5, help="logging step")
    group.add_argument("--dropout", type=float, default=0.2)


def parse_args():
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group(title="General configuration")
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    return parser.parse_args()
