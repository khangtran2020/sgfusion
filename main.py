import os
import torch
import datetime
import warnings
from config import parse_args
from Data.utils import read_data
from utils.console import console
from utils.plotting import plot_learning_curve
from Runs.MCFL.run import run as run_fedsem
from utils.utils import seed_everything, print_args, save_dict, init_history

warnings.filterwarnings("ignore")


def run(args, date, device):

    history = init_history()
    args.name = f"{args.pname}-run-{args.seed}-{args.country}-{date.day}{date.month}-{date.hour}{date.minute}"
    # read data
    with console.status("Initializing Data") as status:
        data_dict = read_data(data_path=args.data_path, country=args.country)
        args.input_dim = 3
        args.output_dim = 1
        args.num_client = len(data_dict.keys())
        console.log(f"Done Reading data: :white_check_mark:")

    if args.mode == "fedsem":
        history = run_fedsem(
            args=args, data_dict=data_dict, device=device, history=history
        )
    save_dict(path=os.path.join("results/dict", f"{args.name}.pkl"), dct=history)
    plot_learning_curve(args=args, history=history)


if __name__ == "__main__":
    date = datetime.datetime.now()
    args = parse_args()
    console.rule(f"Begin experiment: {args.pname}")
    with console.status("Initializing...") as status:
        print_args(args)
        seed_everything(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.log(f"DEVICE USING: {device}")
        if os.path.exists("./results") == False:
            os.mkdir("./results")
        if os.path.exists("./results/model") == False:
            os.mkdir("./results/model")
        if os.path.exists("./results/dict") == False:
            os.mkdir("./results/dict")
        console.log(f"[bold]Done Initializing!")
    # run(args=args, date=date, device=device)
