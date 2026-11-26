"""
Plots learning curves from the runs in data/.

Each run writes data/ppo_<exp_name>_<env>_<date>/log.csv. A curve is the mean (with a
shaded min/max band) over all runs whose exp_name starts with the given prefix, so name
your seeds e.g. q1_baseline_s1, q1_baseline_s2, q1_baseline_s3 and pass `q1_baseline_s`.

Example:
    python plot_results.py --exps q1_nobaseline_s q1_baseline_s --metric eval/return --out q1_return.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_runs(prefix, metric):
    runs = []
    for path in sorted(glob.glob(os.path.join(DATA, f"ppo_{prefix}*", "log.csv"))):
        df = pd.read_csv(path)[["env_steps", metric]].dropna()
        if len(df):
            runs.append(df)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", nargs="+", required=True, help="exp_name prefixes, one curve each")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--metric", default="eval/return")
    parser.add_argument("--out", default="plot.png")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    labels = args.labels or args.exps
    fig, ax = plt.subplots(figsize=(6, 4))
    for prefix, label in zip(args.exps, labels):
        runs = load_runs(prefix, args.metric)
        if not runs:
            print(f"No runs found for prefix '{prefix}'")
            continue
        grid = np.linspace(0, min(r["env_steps"].max() for r in runs), 100)
        ys = np.stack([np.interp(grid, r["env_steps"], r[args.metric]) for r in runs])
        ax.plot(grid, ys.mean(0), label=f"{label} ({len(runs)} runs)")
        ax.fill_between(grid, ys.min(0), ys.max(0), alpha=0.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(args.metric)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_title(args.title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
