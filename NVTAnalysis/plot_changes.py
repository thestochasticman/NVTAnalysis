from json import load
from os import listdir
from pandas import DataFrame
from matplotlib import pyplot as plt

def plot_changes(results_dir: str = 'results'):
    comparisions = []
    for query in listdir(results_dir):
        comparision = load(open(f'{results_dir}/{query}/comparision.json'))
        comparisions += [comparision]
    
    df = DataFrame.from_records(comparisions)
    df = df.set_index("Trial Code")

    ax = df[["experiment_yield", "pre_optimisation_yield", "post_optimisation_yield"]].plot(kind="bar", figsize=(12, 6))
    ax.set_xlabel("Trial Code")
    ax.set_ylabel("Yield")
    ax.set_title("Yields per Trial: Experiment vs Pre vs Post")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("plots/yields_grouped_bar.png", dpi=150)
    # plt.show()

    # 2) Grouped bar chart for errors (pre vs post) per trial
    ax2 = df[["pre_optimisation_error", "post_optimisation_error"]].plot(kind="bar", figsize=(12, 6))
    ax2.set_xlabel("Trial Code")
    ax2.set_ylabel("Error")
    ax2.set_title("Optimisation Error per Trial: Pre vs Post")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("plots/errors_grouped_bar.png", dpi=150)
    # plt.show()

    # 3) Optional: compute deltas and plot as a bar chart to quickly spot improvements
    df["yield_gain"] = df["post_optimisation_yield"] - df["pre_optimisation_yield"]
    df["error_change"] = df["post_optimisation_error"] - df["pre_optimisation_error"]

    ax3 = df[["yield_gain", "error_change"]].plot(kind="bar", figsize=(12, 6))
    ax3.set_xlabel("Trial Code")
    ax3.set_ylabel("Change (Post - Pre)")
    ax3.set_title("Optimisation Changes per Trial: Yield Gain and Error Change")
    plt.xticks(rotation=35, ha="right")
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.savefig("plots/changes_bar.png", dpi=150)
    # plt.show()

if __name__ == '__main__':
    plot_changes()

