import os
from typing import Dict
from argparse import Namespace
import matplotlib.pyplot as plt


def plot_learning_curve(args: Namespace, history: Dict):
    steps = [i * args.eval_step for i in range(len(history["tr_loss"]))]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, history["tr_loss"], label="train", marker="o")
    plt.plot(steps, history["te_liss"], label="test", marker="s")
    plt.ylabel("Root mean square")
    plt.xlabel("Global steps")
    plt.savefig(os.path.join("results/dict", f"{args.name}-results.jpg"))
