from PaddockTS.query import Query
from scipy.optimize import differential_evolution
from daesim2_analysis.parameters import Parameters
from daesim2_analysis.daesim_config import DAESIMConfig
from NVTAnalysis.objective_function import objective_function

def calibrate_dev(
    query: Query,
    daesim_config: DAESIMConfig=DAESIMConfig.from_json_dict('daesim_configs/DAESIM1.json'),
    parameters: Parameters=Parameters.__from_file__('parameters/PARAMS1.json'),
):
    params_info = parameters.df
    params = Parameters.df['Initial Value'].values
    param_bounds =  list(zip(parameters.df["Min"].values, parameters.df["Max"].values))
    result = differential_evolution(
    objective_function,
        bounds = param_bounds,
        args=(parameters.df, [query]),
        popsize=5,
        tol=0.00,
        maxiter=100,
        workers=-1
    )
