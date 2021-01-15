import argparse
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

from mock_metrics import get_mock_data, METRIC_CEILING

# data_directory = Path("../data")
# data_filename = data_directory / "20_11_17_offpolicy_replication_report.xlsx"

# df = pd.read_excel(data_filename, sheet_name=1)

# plot params
colors = ['#F8766D', '#00B0F6']


def plot(domain, intent, ys):
    # fig, axes = plt.subplots(1, len(df.operation.unique()), sharex = 'row', sharey = 'row', figsize = (9, 3))
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    fig.suptitle(f"{domain}:{intent}", size=16)
    metric_labels = ["SRA", "ND_SRA"]

    axes.set_ylabel(f"Accuracy (%)")
    axes.set_xlabel(f"Experiment Run")
    for i, y in enumerate(ys):
        axes.plot(range(y.size), y, color=colors[i], label=metric_labels[i])

    axes.set_ylim([None, METRIC_CEILING])
    axes.legend()
    plt.show()


def plot_multi(data):
    fig, axes = plt.subplots(
        len(data), 1, sharex='col', figsize=(7, 9)
    )
    fig.suptitle(f"SRA and ND_SRA over Experiment Runs", size=16)
    metric_labels = ["SRA", "ND_SRA"]

    for i, (domain, intent, metrics) in enumerate(data):
        axes[i].title.set_text(f"{domain}:{intent}")
        axes[i].title.set_size(12)

        for j, y in enumerate(metrics):
            axes[i].plot(range(y.size), y, color=colors[j],
                         label=metric_labels[j])

        axes[i].set_ylim([None, METRIC_CEILING])
        axes[i].legend()

    axes[math.floor(len(data) / 2)].set_ylabel(f"Accuracy (%)")
    axes[-1].set_xlabel(f"Experiment Run")
    plt.tight_layout()
    plt.show()


def main():
    parser = fetch_parser()
    args = parser.parse_args()

    data = get_mock_data()

    n = min(args.n, 20)

    mpl.style.use('ggplot')
    if args.multigraph:
        plot_multi(data[:n])
    else:
        for (domain, intent, metrics) in data[:n]:
            plot(domain, intent, metrics)


def fetch_parser():
    parser = argparse.ArgumentParser(
        description="Experiment Tracking System Analyzer")
    parser.add_argument('--multigraph', dest='multigraph',
                        action='store_true', help="display graphs on one plot")
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.set_defaults(multigraph=False)
    return parser


if __name__ == "__main__":
    main()
