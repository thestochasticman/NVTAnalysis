# NVTAnalysis

Currently, this package uses experimental data of sowing date, harvest date, yield and the environmental data to calibrate the developmental module of DAESIM.

## DAESIM Development Module

This module is responsible for transition of a crop from one developmental stage to the next.

The developmental stages are as follows

* Vegetative Start, start of vegetation.
* Anthesis or Flowering
* Grainfill. The seed starts growing. 
* Maturity. Seed stops growing, crop starts to die.

## Data

Selected 10 experiments based on rainfall (Low to Medium, Medium to High), yield, region of the site, etc.


## The parameters

```
{
    "Module Path": ["PlantDev", "PlantDev", "PlantDev", "PlantDev", "PlantDev"],
    "Module": ["PlantDev", "PlantDev", "PlantDev", "PlantDev", "PlantDev"],
    "Name": ["gdd_requirements", "gdd_requirements", "gdd_requirements", "gdd_requirements", "gdd_requirements"],
    "Unit": ["deg C d", "deg C d", "deg C d", "deg C d", "deg C d"],
    "Initial Value": [120, 500, 200, 350, 200],
    "Min": [80, 300, 100, 100, 100],
    "Max": [250, 500, 500, 500, 500],
    "Phase Specific": [true, true, true, true, true],
    "Phase": ["germination","vegetative", "anthesis", "grainfill", "maturity"]
}
```

The ranges(Min and Max) are typical for Canola's Hyola Blazer TT sub variety, which is the cultivar we are studying.

## Get Environmental Data

[get_n_representive_sites.py](./NVTAnalysis/get_n_representative_sites.py) contains code to extract the relevant environmental data for the selected sites.

```py
from PaddockTS.Data.environmental import download_environmental_data
from PaddockTS.query import Query
from pandas import read_csv
from os.path import exists
from datetime import date
from datetime import timedelta
from os import remove

def get_n_representative_sites(tmp_dir: str, out_dir: str, reload=False):
    queries = []
    df_Hyola_Blazer_TT = read_csv('data/selected_10_Hyola_Blazer_TT.csv')
    for idx, row in df_Hyola_Blazer_TT.iterrows():
        query = Query(
            lat=row['Trial GPS Lat'],
            lon=row['Trial GPS Long'],
            collections=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
            buffer=0.01,
            bands=[
                'nbart_blue',
                'nbart_green',
                'nbart_red',
                'nbart_red_edge_1',
                'nbart_red_edge_2',
                'nbart_red_edge_3',
                'nbart_nir_1',
                'nbart_nir_2',
                'nbart_swir_2',
                'nbart_swir_3'
            ],
            start_time=date.fromisoformat(row['SowingDate']),
            end_time=date.fromisoformat(row['HarvestDate']) + timedelta(days=50),
            out_dir=out_dir,
            tmp_dir=tmp_dir,
            stub=row['TrialCode']
        )
        queries += [query]
        if not exists(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv') or reload:
            if exists(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv'):
                remove(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv')
            download_environmental_data(query)

```

## Calibration Strategy

The developmental module of DAESIM gives the day of transition for when a crop moves
from one phase to another. But we do not have a ground truth for when those transitions actually happen.

### Canola Stage Sampler

The [Canola Stage Sampler](./NVTAnalysis/canola_stage_sampler_relative.py) uses a dirichlet distribution and prior assumptions of what fraction of the total time
would this cultivar spend in each stage to create observations.


```py
from NVTAnalysis.canola_stage_sampler_relative import CanolaStageSampler

priors = 
sampler = CanolaStageSampler(
    mu_emerg_lag_rel=priors[0],
    mu_veg_lag_rel=priors[1],
    mu_grainfill_lag_rel=priors[2],
    mu_mature_lag_rel=priors[3]
)

## Assuming you created a PaddockTS Query before
priors = np.array([0.05, 0.50, 0.20, 0.20])
synthetic_observables_df = sampler.sample(str(query.start_time), str(query.end_time - timedelta(days=50)), n=100)

print(synthetic_variables)
```

```

emergence_ts          2019-01-13 00:00:00
vegetative_start_ts   2019-01-14 00:00:00
flowering_start_ts    2019-07-10 00:00:00
grainfill_start_ts    2019-09-24 00:00:00
maturity_start_ts     2019-12-25 00:00:00

emergence_doy                          13
vegetative_start_doy                   14
flowering_start_doy                   191
grainfill_start_doy                   267
maturity_start_doy                    359

```
So we are going to try and make sure our model's growth change indices look
as close to the the samplers.

## Objective Function

This is the function we are going to try and minimise

```py
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
        error = ((model_outputs - observations) ** 2) / (observations_unc_sigma**2)
        # error = error[-1]
        errors += [error]
    return sum(errors)/len(errors)
```

In short, it takes the DAESIM parameters, the input query and the priors to create
the target and to generate the model outputs and then we do a simple mean square loss.

## Differential Evolution

```py
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
    priors = (priors / priors.sum()) * 0.95
    params_info = parameters.df
    params = parameters.df['Initial Value'].values
    param_bounds =  list(zip(parameters.df["Min"].values, parameters.df["Max"].values))
    result = differential_evolution(
        objective_function,
        bounds = param_bounds,
        args=(parameters.df, [query], priors),
        popsize=5,
        tol=tol,
        maxiter=100,
        workers=-1,
        seed=123
    )
    
    return result
```



