from PaddockTS.query import Query
from NVTAnalysis.canola_stage_sampler_relative import CanolaStageSampler
from daesim2_analysis.utils import load_df_forcing
from pandas import DataFrame

def get_target_and_uncertainity_from_query(query: Query, sampler: CanolaStageSampler=CanolaStageSampler()):
    observables_names = [
        "start of vegetative",
        "start of flowering",
        "start of grainfill",
        "start of maturity",
        "harvest"
    ]
    observables_units = [
        "ordinal day of year",
        "ordinal day of year",
        "ordinal day of year",
        "ordinal day of year",
        "ordinal day of year"
    ]
    sampler = CanolaStageSampler(seed=123)
    # sampler2 = CanolaWeatherAwareStages(seed=123)
    df=load_df_forcing(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv')
    synthetic_observables_df = sampler.sample(str(query.start_time), str(query.end_time), n=100)
    # synthetic_observables_df = sampler2.extract(df, str(query.start_time), str(query.end_time), n=5)
    observables_values = synthetic_observables_df[
        [
            'vegetative_start_doy',
            'flowering_start_doy',
            'grainfill_start_doy',
            'maturity_start_doy',
            'harvest_doy'
        ]
    ].iloc[0].tolist()
    observables_uncertainty = [5, 5, 5, 5, 5]
    # x = sampler.sample(str(query.start_time), str(query.end_time), n=1)
    target_df = DataFrame({
        "Name": observables_names,
        "Units": observables_units,
        "Values": observables_values,
        "Uncertainty": observables_uncertainty,
    })
    y = target_df["Values"].values
    U_y = target_df["Uncertainty"].values
    return y, U_y, target_df
