import numpy as np
from pandas import DataFrame
from PaddockTS.query import Query
from NVTAnalysis.get_input_data_from_query import get_input_data_from_query
from daesim2_analysis.run import update_attribute, update_attribute_in_phase
from NVTAnalysis.run_model_and_get_outputs import run_model_and_get_outputs

def model_function(params: np.ndarray, params_info: DataFrame, query: Query):
    input_data = get_input_data_from_query(query)
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