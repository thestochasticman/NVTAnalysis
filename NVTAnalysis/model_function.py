import numpy as np
from pandas import DataFrame
from PaddockTS.query import Query
from NVTAnalysis.get_input_data_from_query import get_input_data_from_query
from daesim2_analysis.run import update_attribute, update_attribute_in_phase
from NVTAnalysis.run_model_and_get_outputs import run_model_and_get_outputs

def model_function(params: np.ndarray, params_info: DataFrame, query: Query, training_mode: bool = False):
    input_data = get_input_data_from_query(query, training_mode)
    ODEModelSolver, model_instance, time_axis, forcing_inputs, reset_days, zero_crossing_indices, time_nday_f, time_doy_f, time_year_f = input_data
    for idx, value in enumerate(params):
        param_name = params_info["Name"].values[idx]
        param_path = params_info["Module Path"].values[idx]
        full_path = f"{param_path}.{param_name}"
        phase_specific = params_info["Phase Specific"].values[idx]
        
        if phase_specific:
            # Handle phase-specific parameters
            phase = params_info["Phase"].values[idx]
            update_attribute_in_phase(model_instance, full_path, value, phase)
        else:
            if (param_name == "sowingDays") or (param_name == "harvestDays"):
                # Update parameters that must be defined as a list type
                update_attribute(model_instance, full_path, [value])
            else:
                # Update regular parameters
                update_attribute(model_instance, full_path, value)

        # Make sure the solver knows about the sowing and harvest dates as well (to reset the state variables like GDD and VD)
        if (param_name == "sowingDays") or (param_name == "harvestDays"):
            # Find value of time_nday_f where time_doy_f == sowingDay and time_year_f == sowingYear.
            sowingDay, sowingYear = model_instance.Management.sowingDays, model_instance.Management.sowingYears
            sowing_nday = time_nday_f[(np.floor(time_doy_f) == sowingDay) & (np.array(time_year_f) == sowingYear)]
            
            # Find value of time_nday_f where time_doy_f == sowingDay and time_year_f == sowingYear.
            harvestDay, harvestYear = model_instance.Management.harvestDays, model_instance.Management.harvestYears
            harvest_nday = time_nday_f[(np.floor(time_doy_f) == harvestDay) & (np.array(time_year_f) == harvestYear)]
            
            # Set reset_days to be the updated sowing and harvest nday
            reset_days = [sowing_nday[0], harvest_nday[0]]
     
    model_output = run_model_and_get_outputs(model_instance, ODEModelSolver, time_axis, forcing_inputs, reset_days, zero_crossing_indices)
    return model_output


# --- model_function (replacement) ---
import numpy as np
from pandas import DataFrame
from PaddockTS.query import Query
from NVTAnalysis.get_input_data_from_query import get_input_data_from_query
from daesim2_analysis.run import update_attribute, update_attribute_in_phase
from NVTAnalysis.run_model_and_get_outputs import run_model_and_get_outputs

def model_function(params: np.ndarray, params_info: DataFrame, query: Query, training_mode: bool = False):
    """
    Updates parameters, recomputes reset_days AFTER all updates using nearest matching,
    runs the model, and returns the observable stage transition DOYs.
    """
    # Load inputs
    input_data = get_input_data_from_query(query, training_mode)
    ODEModelSolver, model_instance, time_axis, forcing_inputs, reset_days, zero_crossing_indices, time_nday_f, time_doy_f, time_year_f = input_data

    # ---- 1) Apply all parameter updates (no solver calls here) ----
    for idx, value in enumerate(params):
        param_name = params_info["Name"].values[idx]
        param_path = params_info["Module Path"].values[idx]
        full_path = f"{param_path}.{param_name}"
        phase_specific = params_info["Phase Specific"].values[idx]

        if phase_specific:
            phase = params_info["Phase"].values[idx]
            update_attribute_in_phase(model_instance, full_path, value, phase)
        else:
            if (param_name == "sowingDays") or (param_name == "harvestDays"):
                update_attribute(model_instance, full_path, [value] if not isinstance(value, list) else value)
            else:
                update_attribute(model_instance, full_path, value)

    # ---- 2) Recompute reset_days ONCE, using nearest index (no float equality) ----
    # Coerce mgmt values to scalars
    mgmt = model_instance.Management
    sow_day  = int(np.atleast_1d(mgmt.sowingDays)[0])
    sow_year = int(np.atleast_1d(mgmt.sowingYears)[0])
    har_day  = None
    har_year = None
    if getattr(mgmt, "harvestDays", None) is not None:
        har_day  = int(np.atleast_1d(mgmt.harvestDays)[0])
    if getattr(mgmt, "harvestYears", None) is not None:
        har_year = int(np.atleast_1d(mgmt.harvestYears)[0])

    year_arr = np.asarray(time_year_f)
    doy_arr  = np.floor(np.asarray(time_doy_f)).astype(int)
    nday_arr = np.asarray(time_nday_f)

    # Sowing nearest match within the correct year
    sow_candidates = np.where(year_arr == sow_year)[0]
    if sow_candidates.size == 0:
        # fallback: nearest across all if year not present
        sow_idx = int(np.argmin(np.abs(doy_arr - sow_day)))
    else:
        sow_idx_rel = int(np.argmin(np.abs(doy_arr[sow_candidates] - sow_day)))
        sow_idx = int(sow_candidates[sow_idx_rel])
    sow_nday = float(nday_arr[sow_idx])

    # Harvest nearest (if missing, use end of axis)
    if (har_day is not None) and (har_year is not None):
        har_candidates = np.where(year_arr == har_year)[0]
        if har_candidates.size == 0:
            har_idx = int(np.argmin(np.abs(doy_arr - har_day)))
        else:
            har_idx_rel = int(np.argmin(np.abs(doy_arr[har_candidates] - har_day)))
            har_idx = int(har_candidates[har_idx_rel])
        har_nday = float(nday_arr[har_idx])
        if har_nday < sow_nday:
            # guard against inverted order
            har_nday = float(nday_arr[-1])
    else:
        har_nday = float(nday_arr[-1])

    reset_days = [sow_nday, har_nday]

    # ---- 3) Run and return observable transitions (vegetative, anth0, anth1, maturity, harvest) ----
    model_output = run_model_and_get_outputs(
        model_instance, ODEModelSolver, time_axis, forcing_inputs, reset_days, zero_crossing_indices
    )
    return model_output
