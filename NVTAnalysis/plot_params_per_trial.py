from pandas import read_csv, DataFrame
from json import load
from matplotlib import pyplot as plt
import numpy as np

def get_trial_codes():
    df_Hyola_Blazer_TT = read_csv('data/selected_10_Hyola_Blazer_TT.csv')
    return df_Hyola_Blazer_TT['TrialCode']

def build_paramaters_per_trial_list(trial_codes: list):
    params_per_trial = []
    for trial_code in trial_codes:
        params_file = f'results/{trial_code}/optimised_parameters.json'
        with open(params_file, 'r') as f:
            params_per_trial += [load(f)]
    return params_per_trial


def plot_params_per_trial():
    trial_codes = get_trial_codes()
    params_per_trial = build_paramaters_per_trial_list(trial_codes)

    # Convert to numpy array for easier manipulation
    params_array = np.array(params_per_trial)

    # Phase names corresponding to the 5 parameters
    phase_names = ['Germination', 'Vegetative', 'Anthesis', 'Grainfill', 'Maturity']

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Scatter plot showing all parameters across sites (no implied sequence)
    ax1 = axes[0]
    x_positions = np.arange(len(trial_codes))
    colors = plt.cm.Set2(np.linspace(0, 1, 5))

    for i, phase in enumerate(phase_names):
        ax1.scatter(x_positions, params_array[:, i], label=phase, s=120, alpha=0.7,
                   color=colors[i], edgecolors='black', linewidth=0.5, zorder=3)

    ax1.set_xlabel('Trial Site', fontsize=12)
    ax1.set_ylabel('GDD Requirement (°C d)', fontsize=12)
    ax1.set_title('Optimized GDD Requirements by Trial Site', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(trial_codes, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=1)

    # Plot 2: Grouped bar chart
    ax2 = axes[1]
    bar_width = 0.15
    x_positions = np.arange(len(trial_codes))

    colors = plt.cm.Set3(np.linspace(0, 1, 5))

    for i, phase in enumerate(phase_names):
        offset = (i - 2) * bar_width
        ax2.bar(x_positions + offset, params_array[:, i], bar_width,
                label=phase, alpha=0.8, color=colors[i])

    ax2.set_xlabel('Trial Site', fontsize=12)
    ax2.set_ylabel('GDD Requirement (°C d)', fontsize=12)
    ax2.set_title('Optimized GDD Requirements by Phase and Site (Grouped Bar Chart)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(trial_codes, rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig('plots/params_per_trial.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to plots/params_per_trial.png")
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    df_stats = DataFrame(params_array, columns=phase_names, index=trial_codes)
    print("\nMean GDD requirements across all sites:")
    print(df_stats.mean())
    print("\nStandard deviation:")
    print(df_stats.std())
    print("\nMin values:")
    print(df_stats.min())
    print("\nMax values:")
    print(df_stats.max())


if __name__ == '__main__':
    plot_params_per_trial()