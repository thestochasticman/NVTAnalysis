import numpy as np
import pandas as pd

def run_model_and_get_outputs(Plant, ODEModelSolver, time_axis, forcing_inputs, reset_days, zero_crossing_indices):
    ## Define the callable calculator that defines the right-hand-side ODE function
    PlantCalc = Plant.calculate
    
    Model = ODEModelSolver(calculator=PlantCalc, states_init=[0.0, 0.0], time_start=time_axis[0], log_diagnostics=True)
    
    ## Run the model solver
    res = Model.run(
        time_axis=time_axis,
        forcing_inputs=forcing_inputs,
        solver="euler",
        zero_crossing_indices=zero_crossing_indices,
        reset_days=reset_days,
    )

    # Convert the defaultdict to a regular dictionary
    _diagnostics = dict(Model.diagnostics)
    # Convert each list in the dictionary to a NumPy array
    diagnostics = {key: np.array(value) for key, value in _diagnostics.items()}

    # Convert the array to a numeric type, handling mixed int and float types
    diagnostics['idevphase_numeric'] = np.array(diagnostics['idevphase'],dtype=np.float64)
    
    # In the model idevphase can equal None but that is not useable in post-processing, so we set None values to np.nan
    diagnostics["idevphase_numeric"][diagnostics["idevphase"] == None] = np.nan

    # Add np.nan to the end of each array in the dictionary to represent the last time point in the time_axis (corresponds to the last time point of the state vector)
    for key in diagnostics:
        if key == "t":
            diagnostics[key] = np.append(diagnostics[key], res["t"][-1])
        else:
            diagnostics[key] = np.append(diagnostics[key], np.nan)

    # Add state variables to the diagnostics dictionary
    diagnostics["GDD"] = res["y"][0,:]
    diagnostics["VD"] = res["y"][1,:]

    # Add forcing inputs to diagnostics dictionary
    for i,f in enumerate(forcing_inputs):
        ni = i+1
        if f(time_axis[0]).size == 1:
            fstr = f"forcing {ni:02}"
            diagnostics[fstr] = f(time_axis)
        elif f(time_axis[0]).size > 1:
            # this forcing input has levels/layers (e.g. multilayer soil moisture)
            nz = f(time_axis[0]).size
            for iz in range(nz):
                fstr = f"forcing {ni:02} z{iz}"
                diagnostics[fstr] = f(time_axis)[:,iz]
    
    # Observation Operator
    
    # Diagnose time indexes when developmental phase transitions occur
    ngrowing_seasons = (len(Plant.Management.sowingDays) if (isinstance(Plant.Management.sowingDays, int) == False) else 1)
    if ngrowing_seasons > 1:
        # print("Multiple sowing and harvest events occur. Only returning results for first growing season.")
        ## ignore any time steps before first sowing event and after last harvest event
        it_sowing = np.where(time_axis == reset_days[0])[0][0]  #sowing_steps_itax[0]
        
        if Plant.Management.harvestDays is not None:
            it_harvest = np.where(time_axis == reset_days[1])[0][0]  #harvest_steps_itax[0]   # np.where(np.floor(Climate_doy_f(time_axis)) == Plant.Management.harvestDay)[0][0]
        else:
            it_harvest = -1   # if there is no harvest day specified, we just take the last day of the simulation. 
    else:
        # print("Just one sowing event and one harvest event occurs. Returning results for first (and only) growing season.")
        ## ignore any time steps before first sowing event and after last harvest event
        it_sowing = np.where(time_axis == reset_days[0])[0][0]  #sowing_steps_itax[0]
        
        if Plant.Management.harvestDays is not None:
            it_harvest = np.where(time_axis == reset_days[1])[0][0]  #harvest_steps_itax[0]   # np.where(np.floor(Climate_doy_f(time_axis)) == Plant.Management.harvestDay)[0][0]
        else:
            it_harvest = -1   # if there is no harvest day specified, we just take
    
    # Calculate model-equivalent observations from model run output
    ## Create datetime array from doy and year outputs
    xdoy = np.floor(forcing_inputs[-2](time_axis[it_sowing:it_harvest+1]))
    xyear = np.array(forcing_inputs[-1](time_axis[it_sowing:it_harvest+1]), dtype=int)
    time_index = pd.to_datetime(xyear.astype(str), format='%Y') + pd.to_timedelta(xdoy - 1, unit='D')
    itax_sowing0, itax_mature0, itax_harvest0, itax_phase_transitions0 = Plant.Site.time_index_growing_season(time_index, diagnostics['idevphase_numeric'][it_sowing:it_harvest+1], Plant.Management, Plant.PlantDev, iseason=0)
    itax_sowing = itax_sowing0 + it_sowing
    itax_mature = min(itax_mature0 + it_sowing, it_harvest)
    itax_harvest = min(itax_harvest0 + it_sowing, it_harvest)
    itax_phase_transitions = [min(item + it_sowing, it_harvest) for item in itax_phase_transitions0]
    
    # Developmental phase indexes
    igermination = Plant.PlantDev.phases.index("germination")
    ivegetative = Plant.PlantDev.phases.index("vegetative")
    if Plant.Management.cropType == "Wheat":
        ispike = Plant.PlantDev.phases.index("spike")
    ianthesis = Plant.PlantDev.phases.index("anthesis")
    igrainfill = Plant.PlantDev.phases.index("grainfill")
    imaturity = Plant.PlantDev.phases.index("maturity")

    xdoy = np.floor(forcing_inputs[-2](time_axis))
    xyear = np.array(forcing_inputs[-1](time_axis), dtype=int)
    if ivegetative in diagnostics['idevphase_numeric'][itax_sowing+1:itax_harvest+1]:
        ip = np.where(diagnostics['idevphase'][itax_phase_transitions] == Plant.PlantDev.phases.index('vegetative'))[0][0]
        tdoy_vegetative = xdoy[itax_phase_transitions[ip]]
    else:
        tdoy_anth0 = xdoy[itax_harvest]
    
    if ianthesis in diagnostics['idevphase_numeric'][itax_sowing+1:itax_harvest+1]:
        ip = np.where(diagnostics['idevphase'][itax_phase_transitions] == Plant.PlantDev.phases.index('anthesis'))[0][0]
        tdoy_anth0 = xdoy[itax_phase_transitions[ip]]
    else:
        tdoy_anth0 = xdoy[itax_harvest]
    
    if igrainfill in diagnostics['idevphase_numeric'][itax_sowing+1:itax_harvest+1]:
        ip = np.where(diagnostics['idevphase'][itax_phase_transitions] == Plant.PlantDev.phases.index('grainfill'))[0][0]
        tdoy_anth1 = xdoy[itax_phase_transitions[ip]]
    else:
        tdoy_anth1 = xdoy[itax_harvest]
    
    if imaturity in diagnostics['idevphase_numeric'][itax_sowing+1:itax_harvest+1]:
        ip = np.where(diagnostics['idevphase'][itax_phase_transitions] == Plant.PlantDev.phases.index('maturity'))[0][0]
        tdoy_maturity = time_axis[itax_phase_transitions[ip]]
    

    tdoy_harvest = xdoy[itax_harvest]
    
    # Model output (of observables) given the parameter vector p
    # - this is the model output that we compare to observations and use to calibrate the parameters
    M_p = np.array([
        tdoy_vegetative, 
        tdoy_anth0, 
        tdoy_anth1,
        tdoy_maturity,
        tdoy_harvest,
    ])

    return M_p
