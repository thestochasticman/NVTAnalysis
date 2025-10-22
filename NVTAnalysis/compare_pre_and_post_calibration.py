import numpy as np
from PaddockTS.query import Query
from matplotlib import pyplot as plt
from daesim2_analysis.parameters import Parameters
from NVTAnalysis.calibrate_dev import calibrate_dev
from NVTAnalysis.model_function import model_function
from daesim2_analysis.daesim_config import DAESIMConfig
from NVTAnalysis.get_target_and_uncertainty_from_query import get_target_and_uncertainity_from_query

def plot_observed_vs_initial_vs_optimised_parameters(
        query: Query,
        optimised_parameters: np.ndarray,
        parameters: Parameters,
        results_dir: str,
    ):

    M_x = model_function(optimised_parameters, parameters.df, query)
    M_x0 = model_function(parameters.df['Initial Value'].values, parameters.df, query)
    _, _, target_df = get_target_and_uncertainity_from_query(query)

    fig, axes = plt.subplots(1,1,figsize=(4,3))
    axes.scatter(np.arange(len(target_df)), M_x0, label="Initial Values", color="C0", marker="o", alpha=0.5)
    axes.scatter(np.arange(len(target_df)), M_x, label="Optimized", c="C1", marker="o", alpha=0.5)
    axes.scatter(np.arange(len(target_df)), target_df['Values'], c='k', marker='x', alpha=0.75, label="Observed")
    axes.legend()
    axes.set_xticks(np.arange(M_x.size))
    axes.set_xticklabels(target_df['Name'],rotation=45)
    axes.set_ylabel("Ordinal Day of Year")

    fig.savefig(f'{results_dir}/initial_vs_optimised_vs_observed')


def compare_pre_and_post_calibration(
    query: Query,
    daesim_config: DAESIMConfig = DAESIMConfig.from_json_dict('daesim_configs/DAESIM1.json'),
    parameters: Parameters = Parameters.__from_file__('parameters/PARAMS1.json'),
    results_dir = 'results'
):  
    query_results_dir = f'{results_dir}/{query.stub}'
    result = calibrate_dev(query, daesim_config, parameters)
    optimised_parameters = np.around(np.array(result.x)).astype(int)
    plot_observed_vs_initial_vs_optimised_parameters(query, optimised_parameters, parameters, results_dir)
