import numpy as np
from pandas import DataFrame
from functools import partial
from PaddockTS.query import Query
from NVTAnalysis.model_function import model_function
from NVTAnalysis.get_target_and_uncertainty_from_query import get_target_and_uncertainity_from_query
from NVTAnalysis.canola_stage_sampler_relative import CanolaStageSampler

def objective_function(params: np.ndarray, params_info: DataFrame, queries: list[Query], priors: np.ndarray):
    int_params = np.round(params).astype(int)
    errors = []
    sampler = CanolaStageSampler(
        priors[0],
        priors[1],
        priors[2],
        priors[3]
    )
    for q in queries:
        model_outputs = model_function(int_params, params_info, q, training_mode=True)
        observations, observations_unc_sigma, _ = get_target_and_uncertainity_from_query(q, sampler)
        observations = observations[:-1]
        model_outputs = model_outputs[:-1]
        model_outputs[-1] = model_outputs[-1] + 7
        observations_unc_sigma = observations_unc_sigma[:-1]
        error = np.mean(((model_outputs - observations) ** 2))
        # error = ((model_outputs - observations) ** 2) / (observations_unc_sigma**2)
        # error = error[-1]
        errors += [error]
    return sum(errors)/len(errors)