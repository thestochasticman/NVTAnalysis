import numpy as np
from pandas import DataFrame
from functools import partial
from PaddockTS.query import Query
from NVTAnalysis.model_function import model_function
from NVTAnalysis.get_target_and_uncertainty_from_query import get_target_and_uncertainity_from_query

def calculate_error_for_one_site(query, int_params, param_info):
    model_outputs = model_function(int_params, param_info, query)
    observations, observations_unc_sigma, _ = get_target_and_uncertainity_from_query(query)
    error = np.mean( ((model_outputs - observations) ** 2) / (observations_unc_sigma**2))
    return error

def objective_function(params: np.ndarray, params_info: DataFrame, queries: list[Query]):
    int_params = np.round(params).astype(int)
    errors = []
    calculate_error = partial(calculate_error_for_one_site, int_params=int_params, param_info=params_info)
    for q in queries:
        model_outputs = model_function(int_params, params_info, q)
        observations, observations_unc_sigma, _ = get_target_and_uncertainity_from_query(q)
        error = np.mean( ((model_outputs - observations) ** 2) / (observations_unc_sigma**2))
        error = calculate_error_for_one_site(q, int_params, params_info)
        errors += [error]
    return sum(errors)/len(errors)