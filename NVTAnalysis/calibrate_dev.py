import numpy as np
from PaddockTS.query import Query
from scipy.optimize import differential_evolution
from daesim2_analysis.parameters import Parameters
from daesim2_analysis.daesim_config import DAESIMConfig
from NVTAnalysis.objective_function import objective_function

def calibrate_dev(
    priors: np.ndarray,
    query: Query,
    tol=0,
    daesim_config: DAESIMConfig=DAESIMConfig.from_json_dict('daesim_configs/DAESIM1.json'),
    parameters: Parameters=Parameters.__from_file__('parameters/PARAMS1.json'),
):
    params_info = parameters.df
    params = parameters.df['Initial Value'].values
    param_bounds =  list(zip(parameters.df["Min"].values, parameters.df["Max"].values))
    result = differential_evolution(
        objective_function,
        bounds = param_bounds,
        args=(parameters.df, [query], priors),
        popsize=5,
        tol=0.01,
        maxiter=500,
        workers=-1,
        seed=123
    )
    
    return result

def test():
    from NVTAnalysis.get_n_representative_sites import get_n_representative_sites
    data_dir = '/borevitz_projects/data'
    #data_dir = '/g/data/xe2/ya6227/NVTAnalysis/data/DAESim'
    tmp_dir = data_dir
    out_dir = data_dir
    experiment_df, queries = get_n_representative_sites(tmp_dir=tmp_dir, out_dir=out_dir)
    priors = np.array([0.05181449, 0.47179122, 0.10498441, 0.29451112])
    priors = np.array([0.05, 0.50, 0.20, 0.20])
    priors = np.array([0.08092954, 0.42068518, 0.19754289, 0.26959304, 0.25406048])

    # priors = [0.06582943, 0.52652136, 0.1873288, 0.22032042]
    out = calibrate_dev(priors, queries[3])
    print(out.fun)

if __name__ == '__main__':
    test()
