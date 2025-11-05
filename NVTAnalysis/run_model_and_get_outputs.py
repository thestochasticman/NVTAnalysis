import numpy as np

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
    # Calculate model-equivalent observations from model run output

    # # Diagnose time indexes when developmental phase transitions occur
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
            it_harvest = -1   # if there is no harvest day specified, we just take the last day of the simulation. 

    # Diagnose time indexes when developmental phase transitions occur

    # Convert the array to a numeric type, handling mixed int and float types
    idevphase = diagnostics["idevphase_numeric"]   #[it_sowing:it_harvest+1]
    valid_mask = ~np.isnan(idevphase)
    
    # Identify all transitions (number-to-NaN, NaN-to-number, or number-to-different-number)
    it_phase_transitions = np.where(
        ~valid_mask[:-1] & valid_mask[1:] |  # NaN-to-number
        valid_mask[:-1] & ~valid_mask[1:] |  # Number-to-NaN
        (valid_mask[:-1] & valid_mask[1:] & (np.diff(idevphase) != 0))  # Number-to-different-number
    )[0] + 1
    
    # Time index for the end of the maturity phase
    if Plant.PlantDev.phases.index('maturity') in idevphase:
        it_mature = np.where(idevphase == Plant.PlantDev.phases.index('maturity'))[0][-1]    # Index for end of maturity phase
    elif Plant.Management.harvestDays is not None: 
        it_mature = it_harvest    # Maturity developmental phase not completed, so take harvest as the end of growing season
    else:
        it_mature = -1    # if there is no harvest day specified, we just take the last day of the simulation. 

    # it_sowing = np.where(time_axis == Plant.Management.sowingDay)[0][0]
    # if Plant.Management.harvestDay is not None:
    #     it_harvest = np.where(time_axis == Plant.Management.harvestDay)[0][0]
    # else:
    #     it_harvest = -1   # if there is no harvest day specified, we just take the last day of the simulation. 

    # # Convert the array to a numeric type, handling mixed int and float types
    # idevphase = diagnostics["idevphase_numeric"]
    # valid_mask = ~np.isnan(idevphase)
    
    # # Identify all transitions (number-to-NaN, NaN-to-number, or number-to-different-number)
    # it_phase_transitions = np.where(
    #     ~valid_mask[:-1] & valid_mask[1:] |  # NaN-to-number
    #     valid_mask[:-1] & ~valid_mask[1:] |  # Number-to-NaN
    #     (valid_mask[:-1] & valid_mask[1:] & (np.diff(idevphase) != 0))  # Number-to-different-number
    # )[0] + 1
    
    # # Time index for the end of the maturity phase
    # if Plant.PlantDev.phases.index('maturity') in idevphase:
    #     it_mature = np.where(idevphase == Plant.PlantDev.phases.index('maturity'))[0][-1]    # Index for end of maturity phase
    # elif Plant.Management.harvestDay is not None: 
    #     it_mature = it_harvest    # Maturity developmental phase not completed, so take harvest as the end of growing season
    # else:
    #     it_mature = -1    # if there is no harvest day specified, we just take the last day of the simulation. 

    # import pdb; pdb.set_trace()
    # Filter out transitions that occur on or before the sowing day
    # it_phase_transitions = [t for t in it_phase_transitions if time_axis[t] > time_axis[it_sowing+1]]
    it_phase_transitions = [t for t in it_phase_transitions if t > int(it_sowing+1)]
    # Filter out transitions that occur after the maturity or harvest day
    # it_phase_transitions = [t for t in it_phase_transitions if time_axis[t] <= time_axis[it_mature]]
    it_phase_transitions = [t for t in it_phase_transitions if t <= it_mature]

    # Developmental phase indexes
    igermination = Plant.PlantDev.phases.index("germination")
    ivegetative = Plant.PlantDev.phases.index("vegetative")
    if Plant.Management.cropType == "Wheat":
        ispike = Plant.PlantDev.phases.index("spike")
    ianthesis = Plant.PlantDev.phases.index("anthesis")
    igrainfill = Plant.PlantDev.phases.index("grainfill")
    imaturity = Plant.PlantDev.phases.index("maturity")

    ip = np.where(diagnostics['idevphase'][it_phase_transitions] == Plant.PlantDev.phases.index('vegetative'))[0][0]
    tdoy_vegetative = time_axis[it_phase_transitions[ip]]   # ordinal day-of-year at transition point into vegetative phase
    if Plant.PlantDev.phases.index('anthesis') in idevphase[it_sowing+1:it_harvest+1]:
        ip = np.where(diagnostics['idevphase'][it_phase_transitions] == Plant.PlantDev.phases.index('anthesis'))[0][0]
        tdoy_anth0 = time_axis[it_phase_transitions[ip]]   # ordinal day-of-year at transition point into anthesis phase
    else:
        tdoy_anth0 = time_axis[it_harvest]
    if Plant.PlantDev.phases.index('grainfill') in idevphase[it_sowing+1:it_harvest+1]:
        ip = np.where(diagnostics['idevphase'][it_phase_transitions] == Plant.PlantDev.phases.index('grainfill'))[0][0]
        tdoy_anth1 = time_axis[it_phase_transitions[ip]]   # ordinal day-of-year at transition point into grainfill stage (out of anthesis phase)
    else:
        tdoy_anth1 = time_axis[it_harvest]
    tdoy_harvest = time_axis[it_harvest]   # ordinal day-of-year at harvest
    
    # import pdb; pdb.set_trace()
    # Model output (of observables) given the parameter vector p
    # - this is the model output that we compare to observations and use to calibrate the parameters
    # np.where(diagnostics['idevphase'][it_phase_transitions] == Plant.PlantDev.phases.index('grainfill'))
  
    # print(np.where(diagnostics['idevphase'][it_phase_transitions] == Plant.PlantDev.phases.index('maturity')))
    # try:
    #     ip = np.where(diagnostics['idevphase'][it_phase_transitions] == Plant.PlantDev.phases.index('maturity'))[0][0]
    #     tdoy_maturity = time_axis[it_phase_transitions[ip]]

    # except:
    #     tdoy_maturity = tdoy_harvest + (tdoy_harvest // 4)

    ip = np.where(diagnostics['idevphase'][it_phase_transitions] == Plant.PlantDev.phases.index('maturity'))[0][0]
    tdoy_maturity = time_axis[it_phase_transitions[ip]]

    M_p = np.array([
        tdoy_vegetative, 
        tdoy_anth0, 
        tdoy_anth1, 
        tdoy_maturity,
        tdoy_harvest,
    ])
    return M_p


# import numpy as np

# def run_model_and_get_outputs(Plant, ODEModelSolver, time_axis, forcing_inputs, reset_days, zero_crossing_indices):
#     """
#     Runs the plant ODE model and returns key phenology days:
#       [vegetative_start, anthesis_start, grainfill_start, maturity, harvest]

#     Maturity detection priority:
#       1) Seed/Grain mass plateau (increment stays small for a sustained window)
#       2) Last timestamp where idevphase == 'maturity'
#       3) Harvest day
#     """

#     # ----------------------- Config for yield-based maturity -----------------------
#     _MAT_WINDOW = 3        # steps (e.g., days) to average increment
#     _MAT_ABS_TOL = 0.02    # absolute increment tol per step (units of seed mass per step)
#     _MAT_REL_TOL = 0.002   # relative increment tol per step (fraction per step, e.g. 0.2%)
#     _MAT_CONSEC   = 5      # require this many consecutive steps under both tolerances

#     # Candidate diagnostic keys for seed/grain mass (first found will be used)
#     _SEED_KEYS = ("seed_mass", "yield", "grain_mass", "grainDM", "seedDM", "grain_weight", "seed_weight")

#     # ----------------------- Helper: yield-based maturity detector -----------------
#     def _detect_maturity_by_seed_mass(
#         seed_mass: np.ndarray,
#         it_start: int,
#         it_end: int,
#         window: int = _MAT_WINDOW,
#         abs_tol: float = _MAT_ABS_TOL,
#         rel_tol: float = _MAT_REL_TOL,
#         consecutive: int = _MAT_CONSEC,
#     ) -> int | None:
#         """Return index (in time_axis) of first sustained low-growth point, else None."""
#         if seed_mass is None or it_end <= it_start + window:
#             return None

#         y = np.asarray(seed_mass, dtype=float)
#         if y.size == 0 or np.all(~np.isfinite(y[it_start:it_end+1])):
#             return None

#         # Light interpolation to bridge tiny NaN gaps (conservative)
#         isn = ~np.isfinite(y)
#         if np.any(isn) and np.any(~isn):
#             y[isn] = np.interp(np.flatnonzero(isn), np.flatnonzero(~isn), y[~isn])

#         # Moving average increment over a backward window
#         delta = np.full_like(y, np.nan, dtype=float)
#         t0 = it_start + window
#         for t in range(t0, it_end + 1):
#             delta[t] = (y[t] - y[t - window]) / window

#         eps = 1e-9
#         rel = np.abs(delta) / (np.abs(y) + eps)
#         small = (np.abs(delta) <= abs_tol) & (rel <= rel_tol)

#         run = 0
#         for t in range(t0, it_end + 1):
#             if small[t]:
#                 run += 1
#                 if run >= consecutive:
#                     return t - consecutive + 1
#             else:
#                 run = 0
#         return None

#     # ----------------------- Run model -----------------------
#     PlantCalc = Plant.calculate
#     Model = ODEModelSolver(calculator=PlantCalc, states_init=[0.0, 0.0], time_start=time_axis[0], log_diagnostics=True)

#     res = Model.run(
#         time_axis=time_axis,
#         forcing_inputs=forcing_inputs,
#         solver="euler",
#         zero_crossing_indices=zero_crossing_indices,
#         reset_days=reset_days,
#     )

#     # Diagnostics to ndarray
#     _diagnostics = dict(Model.diagnostics)
#     diagnostics = {key: np.array(value) for key, value in _diagnostics.items()}

#     # Numeric copy of idevphase with None -> NaN
#     diagnostics['idevphase_numeric'] = np.array(diagnostics['idevphase'], dtype=np.float64)
#     diagnostics["idevphase_numeric"][diagnostics["idevphase"] == None] = np.nan  # noqa: E711

#     # Extend diagnostics arrays with a terminal np.nan (or last time for 't')
#     for key in diagnostics:
#         if key == "t":
#             diagnostics[key] = np.append(diagnostics[key], res["t"][-1])
#         else:
#             diagnostics[key] = np.append(diagnostics[key], np.nan)

#     # Add state variables
#     diagnostics["GDD"] = res["y"][0, :]
#     diagnostics["VD"]  = res["y"][1, :]

#     # Add forcing inputs (handles scalar and layered)
#     for i, f in enumerate(forcing_inputs):
#         ni = i + 1
#         sample = f(time_axis[0])
#         if np.size(sample) == 1:
#             fstr = f"forcing {ni:02}"
#             diagnostics[fstr] = f(time_axis)
#         elif np.size(sample) > 1:
#             nz = np.size(sample)
#             vals = f(time_axis)
#             for iz in range(nz):
#                 fstr = f"forcing {ni:02} z{iz}"
#                 diagnostics[fstr] = vals[:, iz]

#     # ----------------------- Sowing/Harvest window -----------------------
#     ngrowing_seasons = (len(Plant.Management.sowingDays) if (isinstance(Plant.Management.sowingDays, int) == False) else 1)
#     if ngrowing_seasons > 1:
#         it_sowing = np.where(time_axis == reset_days[0])[0][0]
#         if Plant.Management.harvestDays is not None:
#             it_harvest = np.where(time_axis == reset_days[1])[0][0]
#         else:
#             it_harvest = -1
#     else:
#         it_sowing = np.where(time_axis == reset_days[0])[0][0]
#         if Plant.Management.harvestDays is not None:
#             it_harvest = np.where(time_axis == reset_days[1])[0][0]
#         else:
#             it_harvest = -1

#     # ----------------------- Phase transitions from idevphase -----------------------
#     idevphase = diagnostics["idevphase_numeric"]
#     valid_mask = ~np.isnan(idevphase)
#     it_phase_transitions = np.where(
#         (~valid_mask[:-1] &  valid_mask[1:]) |  # NaN -> number
#         ( valid_mask[:-1] & ~valid_mask[1:]) |  # number -> NaN
#         ( valid_mask[:-1] &  valid_mask[1:] & (np.diff(idevphase) != 0))  # change
#     )[0] + 1

#     # limit to (sowing+1) ... maturity/harvest (maturity index filled later if needed)
#     it_phase_transitions = [t for t in it_phase_transitions if t > int(it_sowing + 1)]
#     # temp cap by harvest for now; will also cap by maturity once we have it
#     it_phase_transitions = [t for t in it_phase_transitions if t <= it_harvest]

#     # Phase indices
#     igermination = Plant.PlantDev.phases.index("germination")
#     ivegetative  = Plant.PlantDev.phases.index("vegetative")
#     ianthesis    = Plant.PlantDev.phases.index("anthesis")
#     igrainfill   = Plant.PlantDev.phases.index("grainfill")
#     imaturity    = Plant.PlantDev.phases.index("maturity")

#     # Useful phase start helper
#     def _phase_start_idx(phase_idx: int) -> int | None:
#         hits = np.where(idevphase == phase_idx)[0]
#         return int(hits[0]) if hits.size else None

#     it_grainfill_start = _phase_start_idx(igrainfill)
#     it_anthesis_start  = _phase_start_idx(ianthesis)

#     # Key phenology from transitions (with fallbacks)
#     ip = np.where(diagnostics['idevphase'][it_phase_transitions] == ivegetative)[0][0]
#     tdoy_vegetative = time_axis[it_phase_transitions[ip]]

#     if ianthesis in idevphase[it_sowing+1:it_harvest+1]:
#         ip = np.where(diagnostics['idevphase'][it_phase_transitions] == ianthesis)[0][0]
#         tdoy_anth0 = time_axis[it_phase_transitions[ip]]
#     else:
#         tdoy_anth0 = time_axis[it_harvest]

#     if igrainfill in idevphase[it_sowing+1:it_harvest+1]:
#         ip = np.where(diagnostics['idevphase'][it_phase_transitions] == igrainfill)[0][0]
#         tdoy_anth1 = time_axis[it_phase_transitions[ip]]
#     else:
#         tdoy_anth1 = time_axis[it_harvest]

#     tdoy_harvest = time_axis[it_harvest]

#     # ----------------------- Maturity detection -----------------------
#     # 1) yield/seed-mass–based maturity
#     seed_key = next((k for k in _SEED_KEYS if k in diagnostics), None)
#     it_search_start = (
#         it_grainfill_start if it_grainfill_start is not None else
#         it_anthesis_start  if it_anthesis_start  is not None else
#         int(it_sowing + 1)
#     )
#     it_search_end = int(it_harvest)

#     it_mature_yield = None
#     if seed_key is not None:
#         it_mature_yield = _detect_maturity_by_seed_mass(
#             diagnostics[seed_key],
#             it_search_start,
#             it_search_end,
#         )

#     # 2) phase-based maturity (fallback)
#     it_mature_phase = None
#     if imaturity in idevphase:
#         it_mature_phase = np.where(idevphase == imaturity)[0][-1]

#     # 3) final choice
#     if it_mature_yield is not None:
#         it_mature_final = it_mature_yield
#     elif it_mature_phase is not None:
#         it_mature_final = it_mature_phase
#     else:
#         it_mature_final = it_harvest

#     tdoy_maturity = time_axis[it_mature_final]

#     # ----------------------- Return vector -----------------------
#     M_p = np.array([
#         tdoy_vegetative,
#         tdoy_anth0,
#         tdoy_anth1,
#         tdoy_maturity,
#         tdoy_harvest,
#     ])
#     return M_p


# # import numpy as np

# # def run_model_and_get_outputs(Plant, ODEModelSolver, time_axis, forcing_inputs, reset_days, zero_crossing_indices):
# #     """
# #     Returns [vegetative_start, anthesis_start, grainfill_start, maturity, harvest]
# #     Maturity detection (in order of preference):
# #       A) First time running-max(seed_mass) >= 99.9% of final (harvest) yield, AFTER min grainfill days
# #       B) First time weekly Δ running-max(seed_mass) <= 0.005% of final yield (optionally sustained)
# #       C) First time running-max(seed_mass) >= 99.0% of final yield
# #       D) Fallback: harvest
# #     Also records a small diagnostic dict at diagnostics["maturity_meta"].
# #     """

# #     # ----------------------- Tunables -----------------------
# #     FRACTION_OF_FINAL_PRIMARY = 0.999   # 99.9%
# #     FRACTION_OF_FINAL_SECOND  = 0.990   # 99.0% fallback for non-stagnators
# #     WEEK_DAYS                 = 7.0
# #     WEEKLY_FRAC_THRESH        = 0.00005 # 0.005% of final yield per ~week
# #     CONSEC_WEEK_WINDOWS       = 1       # set to 2-3 for more robustness
# #     MIN_GRAINFILL_DAYS        = 5       # don't declare maturity too soon after grainfill

# #     SEED_KEYS = (
# #         "seed_mass", "yield", "grain_mass", "grainDM", "seedDM", "grain_weight", "seed_weight"
# #     )

# #     # ----------------------- Helpers -----------------------
# #     def _interp_fill(a):
# #         a = np.asarray(a, dtype=float).copy()
# #         isn = ~np.isfinite(a)
# #         if np.any(isn) and np.any(~isn):
# #             a[isn] = np.interp(np.flatnonzero(isn), np.flatnonzero(~isn), a[~isn])
# #         return a

# #     def _infer_dt_days(tx: np.ndarray) -> float | None:
# #         dtx = np.diff(np.asarray(tx, dtype=float))
# #         dtx = dtx[dtx > 0]
# #         if dtx.size == 0: return None
# #         dt = float(np.median(dtx))
# #         return dt if np.isfinite(dt) and dt > 0 else None

# #     # ----------------------- Run model -----------------------
# #     PlantCalc = Plant.calculate
# #     Model = ODEModelSolver(calculator=PlantCalc, states_init=[0.0, 0.0], time_start=time_axis[0], log_diagnostics=True)

# #     res = Model.run(
# #         time_axis=time_axis,
# #         forcing_inputs=forcing_inputs,
# #         solver="euler",
# #         zero_crossing_indices=zero_crossing_indices,
# #         reset_days=reset_days,
# #     )

# #     _diagnostics = dict(Model.diagnostics)
# #     diagnostics = {k: np.array(v) for k, v in _diagnostics.items()}

# #     diagnostics['idevphase_numeric'] = np.array(diagnostics['idevphase'], dtype=np.float64)
# #     diagnostics["idevphase_numeric"][diagnostics["idevphase"] == None] = np.nan  # noqa: E711

# #     for key in diagnostics:
# #         if key == "t":
# #             diagnostics[key] = np.append(diagnostics[key], res["t"][-1])
# #         else:
# #             diagnostics[key] = np.append(diagnostics[key], np.nan)

# #     diagnostics["GDD"] = res["y"][0, :]
# #     diagnostics["VD"]  = res["y"][1, :]

# #     for i, f in enumerate(forcing_inputs):
# #         ni = i + 1
# #         sample = f(time_axis[0])
# #         if np.size(sample) == 1:
# #             diagnostics[f"forcing {ni:02}"] = f(time_axis)
# #         elif np.size(sample) > 1:
# #             vals = f(time_axis)
# #             for iz in range(np.size(sample)):
# #                 diagnostics[f"forcing {ni:02} z{iz}"] = vals[:, iz]

# #     # ----------------------- Sowing/Harvest -----------------------
# #     ngrowing_seasons = (len(Plant.Management.sowingDays) if (isinstance(Plant.Management.sowingDays, int) == False) else 1)
# #     if ngrowing_seasons > 1:
# #         it_sowing = np.where(time_axis == reset_days[0])[0][0]
# #         it_harvest = np.where(time_axis == reset_days[1])[0][0] if Plant.Management.harvestDays is not None else -1
# #     else:
# #         it_sowing = np.where(time_axis == reset_days[0])[0][0]
# #         it_harvest = np.where(time_axis == reset_days[1])[0][0] if Plant.Management.harvestDays is not None else -1

# #     idevphase = diagnostics["idevphase_numeric"]
# #     valid_mask = ~np.isnan(idevphase)
# #     it_phase_transitions = np.where(
# #         (~valid_mask[:-1] &  valid_mask[1:]) |
# #         ( valid_mask[:-1] & ~valid_mask[1:]) |
# #         ( valid_mask[:-1] &  valid_mask[1:] & (np.diff(idevphase) != 0))
# #     )[0] + 1
# #     it_phase_transitions = [t for t in it_phase_transitions if t > int(it_sowing + 1)]
# #     it_phase_transitions = [t for t in it_phase_transitions if t <= it_harvest]

# #     ivegetative = Plant.PlantDev.phases.index("vegetative")
# #     ianthesis   = Plant.PlantDev.phases.index("anthesis")
# #     igrainfill  = Plant.PlantDev.phases.index("grainfill")
# #     imaturity   = Plant.PlantDev.phases.index("maturity")

# #     def _phase_start_idx(phase_idx: int) -> int | None:
# #         hits = np.where(idevphase == phase_idx)[0]
# #         return int(hits[0]) if hits.size else None

# #     it_grainfill_start = _phase_start_idx(igrainfill)
# #     it_anthesis_start  = _phase_start_idx(ianthesis)

# #     ip = np.where(diagnostics['idevphase'][it_phase_transitions] == ivegetative)[0][0]
# #     tdoy_vegetative = time_axis[it_phase_transitions[ip]]

# #     if ianthesis in idevphase[it_sowing+1:it_harvest+1]:
# #         ip = np.where(diagnostics['idevphase'][it_phase_transitions] == ianthesis)[0][0]
# #         tdoy_anth0 = time_axis[it_phase_transitions[ip]]
# #     else:
# #         tdoy_anth0 = time_axis[it_harvest]

# #     if igrainfill in idevphase[it_sowing+1:it_harvest+1]:
# #         ip = np.where(diagnostics['idevphase'][it_phase_transitions] == igrainfill)[0][0]
# #         tdoy_anth1 = time_axis[it_phase_transitions[ip]]
# #     else:
# #         tdoy_anth1 = time_axis[it_harvest]

# #     tdoy_harvest = time_axis[it_harvest]

# #     # ----------------------- Maturity (robust) -----------------------
# #     meta = {"rule": None, "it": None, "notes": ""}

# #     seed_key = next((k for k in SEED_KEYS if k in diagnostics), None)
# #     dt_days = _infer_dt_days(time_axis)

# #     # Search window: from grainfill (prefer) or anthesis, else sowing+1
# #     it_search_start = (
# #         it_grainfill_start if it_grainfill_start is not None else
# #         it_anthesis_start  if it_anthesis_start  is not None else
# #         int(it_sowing + 1)
# #     )
# #     it_search_end = int(it_harvest)

# #     # Minimum time after grainfill before maturity can trigger
# #     if dt_days is not None and it_grainfill_start is not None:
# #         min_gf_steps = max(0, int(np.ceil(MIN_GRAINFILL_DAYS / dt_days)))
# #     else:
# #         min_gf_steps = 0

# #     # Phase-based fallback (still useful for metadata)
# #     it_mature_phase = None
# #     if imaturity in idevphase:
# #         it_mature_phase = np.where(idevphase == imaturity)[0][-1]

# #     it_mature_final = it_harvest  # default

# #     if seed_key is not None and dt_days is not None:
# #         y = _interp_fill(diagnostics[seed_key])

# #         # Make series non-decreasing to avoid wiggles: use running max
# #         runmax = np.maximum.accumulate(y)

# #         # Final yield at harvest; if bad, use window max
# #         y_final = runmax[it_harvest] if np.isfinite(runmax[it_harvest]) else np.nan
# #         if not np.isfinite(y_final) or y_final <= 0:
# #             y_final = np.nanmax(runmax[it_search_start:it_search_end+1])

# #         # Primary: 99.9% of final, with min grainfill time
# #         if np.isfinite(y_final) and y_final > 0:
# #             target_A = FRACTION_OF_FINAL_PRIMARY * y_final
# #             sA = max(it_search_start, (it_grainfill_start or it_search_start) + min_gf_steps)
# #             hits_A = np.where(runmax[sA:it_search_end+1] >= target_A)[0]
# #             if hits_A.size:
# #                 itA = sA + int(hits_A[0])
# #                 it_mature_final = itA
# #                 meta.update({"rule": "A_99.9pct_final", "it": int(itA)})

# #             else:
# #                 # Secondary: weekly Δ ≤ 0.005% of final (sustained)
# #                 k = max(1, int(round(WEEK_DAYS / dt_days)))
# #                 abs_week_thresh = WEEKLY_FRAC_THRESH * y_final
# #                 run = 0
# #                 itB = None
# #                 for t in range(max(it_search_start + k, (it_grainfill_start or it_search_start) + min_gf_steps), it_search_end + 1):
# #                     dy = runmax[t] - runmax[t - k]
# #                     if np.isfinite(dy) and dy <= abs_week_thresh:
# #                         run += 1
# #                         if run >= CONSEC_WEEK_WINDOWS:
# #                             itB = t - (CONSEC_WEEK_WINDOWS - 1)
# #                             break
# #                     else:
# #                         run = 0
# #                 if itB is not None:
# #                     it_mature_final = itB
# #                     meta.update({"rule": "B_weekly_delta", "it": int(itB), "notes": f"k={k}, thr={abs_week_thresh:.3g}"})
# #                 else:
# #                     # Tertiary: 99.0% of final (for strong non-stagnators)
# #                     target_C = FRACTION_OF_FINAL_SECOND * y_final
# #                     sC = max(it_search_start, (it_grainfill_start or it_search_start) + min_gf_steps)
# #                     hits_C = np.where(runmax[sC:it_search_end+1] >= target_C)[0]
# #                     if hits_C.size:
# #                         itC = sC + int(hits_C[0])
# #                         it_mature_final = itC
# #                         meta.update({"rule": "C_99.0pct_final", "it": int(itC)})
# #                     else:
# #                         # Final fallback: harvest (or phase if you prefer)
# #                         if it_mature_phase is not None:
# #                             it_mature_final = it_mature_phase
# #                             meta.update({"rule": "D_phase_fallback", "it": int(it_mature_phase)})
# #                         else:
# #                             it_mature_final = it_harvest
# #                             meta.update({"rule": "E_harvest_fallback", "it": int(it_harvest)})
# #     else:
# #         # No seed_mass series or no dt; use phase/harvest
# #         if it_mature_phase is not None:
# #             it_mature_final = it_mature_phase
# #             meta.update({"rule": "D_phase_fallback", "it": int(it_mature_phase)})
# #         else:
# #             it_mature_final = it_harvest
# #             meta.update({"rule": "E_harvest_fallback", "it": int(it_harvest)})

# #     tdoy_maturity = time_axis[it_mature_final]
# #     diagnostics["maturity_meta"] = meta  # small breadcrumb for auditing

# #     # ----------------------- Return -----------------------
# #     M_p = np.array([
# #         tdoy_vegetative,
# #         tdoy_anth0,
# #         tdoy_anth1,
# #         tdoy_maturity,
# #         tdoy_harvest,
# #     ])
# #     return M_p
