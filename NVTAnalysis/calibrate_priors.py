
from PaddockTS.query import Query
from NVTAnalysis.calibrate_dev import calibrate_dev
from NVTAnalysis.compare_pre_and_post_calibration import compare_pre_and_post_calibration
from scipy.optimize import differential_evolution
from daesim2_analysis.daesim_config import DAESIMConfig
from daesim2_analysis.parameters import Parameters
from pandas import DataFrame
import numpy as np

def objective_function(priors, query, experiment_df, daesim_config, parameters, results_dir, optimisation_mode):
    priors = priors/priors.sum()
    # return compare_pre_and_post_calibration(
    #     priors,
    #     query,
    #     daesim_config,
    #     parameters,
    #     results_dir,
    #     prior_optimisation_mode=True
    # )
    return compare_pre_and_post_calibration(
        priors=priors,
        query=query,
        experiment_df=experiment_df,
        daesim_config=daesim_config,
        parameters=parameters,
        results_dir=results_dir,
        prior_optimisation_mode=True
    )

def calibrate_priors(
        query: Query,
        experiment_df: DataFrame,
        daesim_config: DAESIMConfig = DAESIMConfig.from_json_dict('daesim_configs/DAESIM1.json'),
        parameters: Parameters = Parameters.__from_file__('parameters/PARAMS1.json'),

        )->np.ndarray:
    priors = np.array([0.05, 0.50, 0.20, 0.20])
    # bounds = np.array([[0.03, 0.10], [0.30, 0.60], [0.10, 0.30], [0.10, 0.30]])
    bounds = np.array([[0.03, 0.10], [0.25, 0.4], [0.05, 0.25], [0.2, 0.30], [0, 0.30]])

    
    result = differential_evolution(
        # compare_pre_and_post_calibration,
        objective_function,
        bounds = bounds,
        args=(query, experiment_df, daesim_config, parameters, 'results', True),
        popsize=1,
        tol=0.2,
        maxiter=10,
        workers=1,
        seed=123
    )


def test():
    from NVTAnalysis.get_n_representative_sites import get_n_representative_sites
    data_dir = '/borevitz_projects/data'
    #data_dir = '/g/data/xe2/ya6227/NVTAnalysis/data/DAESim'
    tmp_dir = data_dir
    out_dir = data_dir
    experiment_df, queries = get_n_representative_sites(tmp_dir=tmp_dir, out_dir=out_dir)
    calibrate_priors(queries[5], experiment_df)
if __name__ == '__main__':
    test()
