import numpy as np
from PaddockTS.query import Query
from matplotlib import pyplot as plt
from daesim2_analysis.parameters import Parameters
from NVTAnalysis.calibrate_dev import calibrate_dev
from NVTAnalysis.model_function import model_function
from daesim2_analysis.daesim_config import DAESIMConfig
from NVTAnalysis.get_target_and_uncertainty_from_query import get_target_and_uncertainity_from_query
from daesim2_analysis.run import update_and_run_model
from daesim2_analysis.utils import load_df_forcing
from daesim2_analysis.experiment import Experiment
from os import makedirs

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

def plot_model_run(
        dev_params: np.ndarray,
        experiment: Experiment,
        plot_destination: str,
    ):
    d_fd_mapping = {
        'Climate_solRadswskyb_f': 'forcing 01',
        'Climate_solRadswskyd_f': 'forcing 02',
        'Climate_airTempCMin_f': 'forcing 03',
        'Climate_airTempCMax_f': 'forcing 04',
        'Climate_airPressure_f': 'forcing 05',
        'Climate_airRH_f': 'forcing 06',
        'Climate_airCO2_f': 'forcing 07',
        'Climate_airO2_f': 'forcing 08',
        'Climate_airU_f': 'forcing 09',
        'Climate_soilTheta_z_f': 'forcing 10',
        'Climate_doy_f': 'forcing 11',
        'Climate_year_f': 'forcing 12'
    }

    parameters = experiment.parameters

    model_output = update_and_run_model(
        dev_params,
        # [120, 500, 200, 350, 200],
        # [205, 492, 298, 135, 393],
        experiment.PlantX,
        experiment.input_data,
        parameters.df,
        parameters.problem,

    )

    fig, axes = plt.subplots(5,1,figsize=(8,10),sharex=True)
    axes[0].plot(model_output['t'], model_output["LAI"])
    axes[0].set_ylabel("LAI\n"+r"($\rm m^2 \; m^{-2}$)")
    axes[0].tick_params(axis='x', labelrotation=45)
    axes[0].annotate("Leaf area index", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[0].set_ylim([0,6.5])

    axes[1].plot(model_output["t"], model_output["GPP"])
    axes[1].set_ylabel("GPP\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
    axes[1].tick_params(axis='x', labelrotation=45)
    axes[1].annotate("Photosynthesis", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[1].set_ylim([0,30])

    axes[2].plot(model_output["t"], model_output["E_mmd"])
    axes[2].set_ylabel(r"$\rm E$"+"\n"+r"($\rm mm \; d^{-1}$)")
    axes[2].tick_params(axis='x', labelrotation=45)
    axes[2].annotate("Transpiration Rate", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[2].set_ylim([0,6])

    axes[3].plot(model_output["t"], model_output["Bio_time"])
    axes[3].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C d)")
    axes[3].annotate("Growing Degree Days", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)

    alp = 0.6
    axes[4].plot(model_output["t"], model_output["Cleaf"]+model_output["Croot"]+model_output["Cstem"]+model_output["Cseed"],c='k',label="Plant", alpha=alp)
    axes[4].plot(model_output["t"], model_output["Cleaf"],label="Leaf", alpha=alp)
    axes[4].plot(model_output["t"], model_output["Cstem"],label="Stem", alpha=alp)
    axes[4].plot(model_output["t"], model_output["Croot"],label="Root", alpha=alp)
    axes[4].plot(model_output["t"], model_output["Cseed"],label="Seed", alpha=alp)
    axes[4].set_ylabel("Carbon Pool Size\n"+r"(g C $\rm m^{-2}$)")
    axes[4].set_xlabel("Time (day of year)")
    axes[4].legend(loc=3,fontsize=9,handlelength=0.8)

    # Time indexing for model output data, to determine outputs at specific times in the growing season
    itax_sowing, itax_mature, itax_harvest, itax_phase_transitions = experiment.PlantX.Site.time_index_growing_season(experiment.ForcingDataX.time_index, model_output['idevphase_numeric'], experiment.PlantX.Management, experiment.PlantX.PlantDev)
    harvest_index_maturity = model_output["Cseed"][itax_harvest] / (model_output["Cleaf"][itax_mature]+model_output["Croot"][itax_mature]+model_output["Cstem"][itax_mature])
    yield_from_seed_Cpool = model_output["Cseed"][itax_harvest]/100 * (1/experiment.PlantX.PlantCH2O.f_C)   ## convert gC m-2 to t dry biomass ha-1
    axes[4].annotate("Yield = %1.2f t/ha" % (yield_from_seed_Cpool), (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[4].annotate("Harvest index = %1.2f" % (harvest_index_maturity), (0.01,0.81), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[0].set_xlim([experiment.PlantX.Management.sowingDays[0], model_output[d_fd_mapping['Climate_doy_f']][-1]])
    plt.tight_layout()
    fig.savefig(plot_destination)


def compare_pre_and_post_calibration(
    query: Query,
    daesim_config: DAESIMConfig = DAESIMConfig.from_json_dict('daesim_configs/DAESIM1.json'),
    parameters: Parameters = Parameters.__from_file__('parameters/PARAMS1.json'),
    results_dir = 'results'
):  
    query_results_dir = f'{results_dir}/{query.stub}'
    makedirs(query_results_dir, exist_ok=True)
    result = calibrate_dev(query, daesim_config, parameters)
    optimised_parameters = np.around(np.array(result.x)).astype(int)
    plot_observed_vs_initial_vs_optimised_parameters(query, optimised_parameters, parameters, query_results_dir)

    experiment = Experiment(
        crop_type='Canola',
        CLatDeg=query.lat,
        CLonDeg=query.lon,
        sowing_dates=[query.start_time],
        harvest_dates=[query.end_time],
        df_forcing=load_df_forcing(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv'),
        xsite=query.stub,
        daesim_config=daesim_config,
        parameters='parameters/PARAMS2.json',
    )

    plot_model_run(
        optimised_parameters,
        experiment,
        plot_destination=f'{query_results_dir}/optimised_dev_params_run'
    )

    plot_model_run(
        [120, 500, 200, 350, 200],
        experiment,
        plot_destination=f'{query_results_dir}/original_dev_params_run'
    )

