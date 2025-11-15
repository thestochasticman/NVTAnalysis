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
from NVTAnalysis.canola_stage_sampler_relative import CanolaStageSampler
from pandas import DataFrame
from json import dump
from os import makedirs
from datetime import timedelta
import pandas as pd

def plot_observed_vs_initial_vs_optimised_parameters(
        priors: np.ndarray,
        query: Query,
        optimised_parameters: np.ndarray,
        parameters: Parameters,
        results_dir: str,
    ):

    M_x = model_function(optimised_parameters, parameters.df, query)
    M_x0 = model_function(parameters.df['Initial Value'].values, parameters.df, query)
    sampler = CanolaStageSampler(
        priors[0],
        priors[1],
        priors[2],
        priors[3],
    )
    _, _, target_df = get_target_and_uncertainity_from_query(query, sampler)

    fig, axes = plt.subplots(1,1,figsize=(4,3))
    axes.scatter(np.arange(len(target_df)), M_x0, label="Initial Values", color="C0", marker="o", alpha=0.5)
    axes.scatter(np.arange(len(target_df)), M_x, label="Optimized", c="C1", marker="o", alpha=0.5)
    axes.scatter(np.arange(len(target_df)), target_df['Values'], c='k', marker='x', alpha=0.75, label="Observed")
    axes.legend()
    axes.set_xticks(np.arange(M_x.size))
    axes.set_xticklabels(target_df['Name'],rotation=45)
    axes.set_ylabel("Ordinal Day of Year")

    fig.savefig(f'{results_dir}/initial_vs_optimised_vs_observed')
    
    return M_x, M_x0

def plot_model_run(
        stages: np.ndarray,
        dev_params: np.ndarray,
        experiment: Experiment,
        plot_destination: str,

    ):
    stages = stages[:-1]
    print(stages)
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



    iseason=0
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

    itax_sowing, itax_mature, itax_harvest, itax_phase_transitions = experiment.PlantX.Site.time_index_growing_season(experiment.ForcingDataX.time_index, model_output['idevphase_numeric'], experiment.PlantX.Management, experiment.PlantX.PlantDev)
    harvest_index_maturity = model_output["Cseed"][itax_harvest] / (model_output["Cleaf"][itax_mature]+model_output["Croot"][itax_mature]+model_output["Cstem"][itax_mature])
    yield_from_seed_Cpool = model_output["Cseed"][itax_harvest]/100 * (1/experiment.PlantX.PlantCH2O.f_C)   ## convert gC m-2 to t dry biomass ha-1
    axes[4].annotate("Yield = %1.2f t/ha" % (yield_from_seed_Cpool), (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[4].annotate("Harvest index = %1.2f" % (harvest_index_maturity), (0.01,0.81), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[0].set_xlim([experiment.PlantX.Management.sowingDays[0], model_output[d_fd_mapping['Climate_doy_f']][-1]])


    phase_labels = ['vegetative start', 'anthesis', 'grainfill', 'maturity']


    stage_times = np.asarray(stages, dtype=float).ravel()


    seen = set()
    stage_times = [t for t in stage_times if np.isfinite(t) and (t not in seen and not seen.add(t))]

    xmin, xmax = axes[0].get_xlim()
    stage_times = [t for t in stage_times if xmin <= t <= xmax]


    for ax in axes:
        for t in stage_times:
            ax.axvline(t, linestyle='--', linewidth=1.0, alpha=0.7)

  
    labels = phase_labels[:len(stage_times)]
    for t, lbl in zip(stage_times, labels):
        for ax in axes:
            ax.annotate(
                lbl, xy=(t, 0.5), xycoords=('data', 'axes fraction'),
                xytext=(0, 5), textcoords='offset points',
                rotation=0, ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7)
            )



    plt.tight_layout()
    fig.savefig(plot_destination)
    plt.close(fig)
    return yield_from_seed_Cpool




def plot_model_run(
        stages: np.ndarray,
        dev_params: np.ndarray,
        experiment: Experiment,
        plot_destination: str,

    ):
    iseason=0
    stages = stages[:-1]
    print(stages)
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
        experiment.PlantX,
        experiment.input_data,
        parameters.df,
        parameters.problem,

    )

    ## Create datetime array from doy and year outputs
    xyear = np.array(model_output[d_fd_mapping['Climate_year_f']], dtype=int)
    xdoy = model_output[d_fd_mapping['Climate_doy_f']]
    model_output['Date'] = pd.to_datetime(xyear.astype(str), format='%Y') + pd.to_timedelta(xdoy - 1, unit='D')

    ## Create figure
    fig, axes = plt.subplots(5,1,figsize=(8,10),sharex=True)
    axes[0].plot(model_output['Date'], model_output["LAI"])
    axes[0].set_ylabel("LAI\n"+r"($\rm m^2 \; m^{-2}$)")
    axes[0].tick_params(axis='x', labelrotation=45)
    axes[0].annotate("Leaf area index", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[0].set_ylim([0,6.5])

    axes[1].plot(model_output["Date"], model_output["GPP"])
    axes[1].set_ylabel("GPP\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
    axes[1].tick_params(axis='x', labelrotation=45)
    axes[1].annotate("Photosynthesis", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[1].set_ylim([0,30])

    axes[2].plot(model_output["Date"], model_output["E_mmd"])
    axes[2].set_ylabel(r"$\rm E$"+"\n"+r"($\rm mm \; d^{-1}$)")
    axes[2].tick_params(axis='x', labelrotation=45)
    axes[2].annotate("Transpiration Rate", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[2].set_ylim([0,6])

    axes[3].plot(model_output["Date"], model_output["Bio_time"])
    axes[3].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C d)")
    axes[3].annotate("Growing Degree Days", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)

    alp = 0.6
    axes[4].plot(model_output["Date"], model_output["Cleaf"]+model_output["Croot"]+model_output["Cstem"]+model_output["Cseed"],c='k',label="Plant", alpha=alp)
    axes[4].plot(model_output["Date"], model_output["Cleaf"],label="Leaf", alpha=alp)
    axes[4].plot(model_output["Date"], model_output["Cstem"],label="Stem", alpha=alp)
    axes[4].plot(model_output["Date"], model_output["Croot"],label="Root", alpha=alp)
    axes[4].plot(model_output["Date"], model_output["Cseed"],label="Seed", alpha=alp)
    axes[4].set_ylabel("Carbon Pool Size\n"+r"(g C $\rm m^{-2}$)")
    axes[4].set_xlabel("Time (day of year)")
    axes[4].legend(loc=3,fontsize=9,handlelength=0.8)

    itax_sowing, itax_mature, itax_harvest, itax_phase_transitions = experiment.PlantX.Site.time_index_growing_season(experiment.ForcingDataX.time_index, model_output['idevphase_numeric'], experiment.PlantX.Management, experiment.PlantX.PlantDev)
    print(itax_sowing, itax_mature, itax_harvest, itax_phase_transitions)
    harvest_index_maturity = model_output["Cseed"][itax_harvest] / (model_output["Cleaf"][itax_mature]+model_output["Croot"][itax_mature]+model_output["Cstem"][itax_mature])
    yield_from_seed_Cpool = model_output["Cseed"][itax_harvest]/100 * (1/experiment.PlantX.PlantCH2O.f_C)   ## convert gC m-2 to t dry biomass ha-1
    axes[4].annotate("Yield = %1.2f t/ha" % (yield_from_seed_Cpool), (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    axes[4].annotate("Harvest index = %1.2f" % (harvest_index_maturity), (0.01,0.81), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
    # axes[0].set_xlim([experiment.PlantX.Management.sowingDays[0], model_output[d_fd_mapping['Climate_doy_f']][-1]])


    # Add annotations for developmental phases
    itax_sowing, itax_mature, itax_harvest, itax_phase_transitions = experiment.PlantX.Site.time_index_growing_season(experiment.ForcingDataX.time_index, model_output['idevphase_numeric'], experiment.PlantX.Management, experiment.PlantX.PlantDev, iseason=iseason)
    print(itax_sowing, itax_mature, itax_harvest, itax_phase_transitions)
    # for ax in axes:
    ax = axes[3]
    ylimmin, ylimmax = 0, ax.get_ylim()[1]
    xlimmin = model_output['Date'][0]
    xlimmax = model_output['Date'][-1]
    for itime in itax_phase_transitions:
        ax.vlines(x=model_output['Date'][itime], ymin=ylimmin, ymax=ylimmax, color='0.5',linestyle="--")
        text_x = model_output['Date'][itime] + pd.Timedelta(days=1)
        text_y = 0.5 * ylimmax
        if (text_x < xlimmin) or (text_x > xlimmax):
            continue
        elif ~np.isnan(model_output['idevphase_numeric'][itime]):
            # print(experiment.PlantX.PlantDev.phases)
            # print(model_output['idevphase'][itime])
            phase = experiment.PlantX.PlantDev.phases[int(model_output['idevphase'][itime])]
            ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='center',
                    fontsize=8, alpha=0.7, rotation=90)
        elif np.isnan(model_output['idevphase_numeric'][itime]):
            phase = "mature"
            ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='center',
                    fontsize=8, alpha=0.7, rotation=90)
    ax.set_ylim([ylimmin, ylimmax])

    # Alex note: I have replaced the code below with the code above, which plots all stage transition points, not
    # just the ones in 'stages'
    # phase_labels = ['vegetative start', 'anthesis', 'grainfill', 'maturity']


    # stage_times = np.asarray(stages, dtype=float).ravel()


    # seen = set()
    # stage_times = [t for t in stage_times if np.isfinite(t) and (t not in seen and not seen.add(t))]

    # xmin, xmax = axes[0].get_xlim()
    # stage_times = [t for t in stage_times if xmin <= t <= xmax]


    # for ax in axes:
    #     for t in stage_times:
    #         ax.axvline(t, linestyle='--', linewidth=1.0, alpha=0.7)


    # import pdb; pdb.set_trace()
    # labels = phase_labels[:len(stage_times)]
    # for t, lbl in zip(stage_times, labels):
    #     for ax in axes:
    #         ax.annotate(
    #             lbl, xy=(t, 0.5), xycoords=('data', 'axes fraction'),
    #             xytext=(0, 5), textcoords='offset points',
    #             rotation=0, ha='center', va='bottom', fontsize=9,
    #             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7)
    #         )



    plt.tight_layout()
    fig.savefig(plot_destination)
    plt.close(fig)
    return yield_from_seed_Cpool

def only_get_yield_from_model(
    dev_params: np.ndarray,
    experiment: Experiment
): 
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
    itax_sowing, itax_mature, itax_harvest, itax_phase_transitions = experiment.PlantX.Site.time_index_growing_season(experiment.ForcingDataX.time_index, model_output['idevphase_numeric'], experiment.PlantX.Management, experiment.PlantX.PlantDev)
    yield_from_seed_Cpool = model_output["Cseed"][itax_harvest]/100 * (1/experiment.PlantX.PlantCH2O.f_C)
    return yield_from_seed_Cpool

def compare_pre_and_post_calibration(
    priors: np.ndarray,
    query: Query,
    experiment_df: DataFrame,
    daesim_config: DAESIMConfig = DAESIMConfig.from_json_dict('daesim_configs/DAESIM1.json'),
    parameters: Parameters = Parameters.__from_file__('parameters/PARAMS1.json'),
    results_dir = 'results',
    prior_optimisation_mode = False
):  
    # priors = (priors / priors.sum()) * 0.95
    query_results_dir = f'{results_dir}/{query.stub}'
    makedirs(query_results_dir, exist_ok=True)
    if not prior_optimisation_mode:
        result = calibrate_dev(priors, query, 0, daesim_config, parameters)
    else:
        result = calibrate_dev(priors, query, 0, daesim_config, parameters)
    optimised_parameters = np.around(np.array(result.x)).astype(int)
    dump(optimised_parameters.tolist(), open(f'{query_results_dir}/optimised_parameters.json', 'w'))
    if not prior_optimisation_mode:
        stages_optimised, stages_initial = plot_observed_vs_initial_vs_optimised_parameters(priors, query, optimised_parameters, parameters, query_results_dir)
    
    experiment = Experiment(
        crop_type='Canola',
        CLatDeg=query.lat,
        CLonDeg=query.lon,
        sowing_dates=[query.start_time],
        harvest_dates=[query.end_time - timedelta(days=50)] ,
        df_forcing=load_df_forcing(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv'),
        xsite=query.stub,
        daesim_config=daesim_config,
        parameters='parameters/PARAMS1.json',
    )

    if not prior_optimisation_mode:
        post_opimisation_yield = plot_model_run(
            stages_optimised,
            optimised_parameters,
            experiment,
            plot_destination=f'{query_results_dir}/optimised_dev_params_run'
        )

        pre_optimisation_yield = plot_model_run(
            stages_initial,
            experiment.parameters.init,
            experiment,
            plot_destination=f'{query_results_dir}/original_dev_params_run'
        )
    
    else:
        post_opimisation_yield = only_get_yield_from_model(
            optimised_parameters,
            experiment
        )

        pre_optimisation_yield = only_get_yield_from_model(
            experiment.parameters.init,
            experiment
        )

    experiment_row = experiment_df[experiment_df['TrialCode'] == query.stub]
    experiment_yield = experiment_row['Single Site Yield'].iloc[0]
    
    if not prior_optimisation_mode:
        comparision = {
            'pre_optimisation_error': pre_optimisation_yield  - experiment_yield,
            'post_optimisation_error': post_opimisation_yield - experiment_yield,
            'experiment_yield': experiment_yield,
            'pre_optimisation_yield': pre_optimisation_yield,
            'post_optimisation_yield': post_opimisation_yield,
            'Trial Code': query.stub
        }

        dump(comparision, open(f'{query_results_dir}/comparision.json', 'w+'))

    print(priors, post_opimisation_yield - experiment_yield, query.stub)
    return (post_opimisation_yield - experiment_yield)**2


def test():
    from NVTAnalysis.get_n_representative_sites import get_n_representative_sites
    data_dir = '/borevitz_projects/data'
    #data_dir = '/g/data/xe2/ya6227/NVTAnalysis/data/DAESim'
    tmp_dir = data_dir
    out_dir = data_dir
    experiment_df, queries = get_n_representative_sites(tmp_dir=tmp_dir, out_dir=out_dir)
    priors = np.array([0.03535086, 0.56213529, 0.22668816, 0.17582569])
    priors =np.array([0.06582943, 0.52652136, 0.1873288, 0.22032042])
    # priors = np.array([0.07028798, 0.48785009, 0.23681045, 0.20505148])
    # priors = np.array([0.06278367, 0.52598781, 0.2698857, 0.14134282])
    # priors = np.array([0.03422608, 0.41178264, 0.27057179, 0.28341949])
    priors = np.array([0.06437209, 0.54745734, 0.10846586, 0.27970471])
    priors = np.array([0.05, 0.40, 0.10, 0.40])
    # priors = np.array([0.05, 0.38, 0.07, 0.45])
    # priors = np.array([0.05, 0.50, 0.20, 0.20])
    # priors = np.array([0.05, 0.36, 0.04, 0.50])
    priors = np.array([10/109, 48/109, 18/109, 23/109])
    # print([priors])
    for query in queries:
        compare_pre_and_post_calibration(priors, query, experiment_df)
    # compare_pre_and_post_calibration(priors, queries[3], experiment_df)

if __name__ == '__main__':
    test()
