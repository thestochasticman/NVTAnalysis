from PaddockTS.query import Query
from daesim.climate import *
from daesim.plantgrowthphases import PlantGrowthPhases
from daesim.management import ManagementModule
from daesim.plant_1000_thermaltime import PlantModuleCalculator
from pandas import Timestamp
from daesim2_analysis.utils import load_df_forcing
from daesim2_analysis.forcing_data import ForcingData
from daesim.utils import ODEModelSolver

def get_input_data_from_query(query: Query):
    SiteX = ClimateModule(CLatDeg=query.lat,CLonDeg=query.lon,timezone=10)
    ForcingDataX = ForcingData(
        SiteX=SiteX,
        sowing_dates=[Timestamp(query.start_time)],
        harvest_dates=[Timestamp(query.end_time)],
        df=load_df_forcing(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv'),
        df_type='0',
        zero_crossing_indices=[0,1]
    )
    ManagementX = ManagementModule(cropType="Canola", sowingDays=ForcingDataX.sowing_days, harvestDays=ForcingDataX.harvest_days, sowingYears=ForcingDataX.sowing_years, harvestYears=ForcingDataX.harvest_years)
    PlantDevX = PlantGrowthPhases(
        phases=["germination", "vegetative", "anthesis", "grainfill", "maturity"],
        gdd_requirements=[120, 500, 200, 350, 200], #rid this
        vd_requirements=[0, 25, 0, 0, 0], #keep this
        allocation_coeffs=[ #rid this
            [0.2, 0.1, 0.7, 0.0, 0.0],   # Phase 1
            [0.5, 0.1, 0.4, 0.0, 0.0],   # Phase 2
            [0.25, 0.5, 0.25, 0.0, 0.0], # Phase 3
            [0.1, 0.1, 0.1, 0.7, 0.0],   # Phase 4
            [0.1, 0.1, 0.1, 0.7, 0.0]    # Phase 5
        ],
        turnover_rates = [ #rid this
            [0.001, 0.001, 0.001, 0.0, 0.0],  # Phase 1
            [0.01,  0.002, 0.01,  0.0, 0.0],  # Phase 2
            [0.02,  0.002, 0.04,  0.0, 0.0],  # Phase 3
            [0.10,  0.008, 0.10,  0.0, 0.0],  # Phase 4
            [0.50,  0.017, 0.50,  0.0, 0.0]   # Phase 5
        ]    ## Turnover rates per pool and developmental phase (days-1))
    )
    
    PlantX = PlantModuleCalculator(
        Site=SiteX,
        Management=ManagementX,
        PlantDev=PlantDevX,
        GDD_method="linear1",
        GDD_Tbase=0.0,
        GDD_Tupp=25.0,
    )
    
    # %%
    ## Define the callable calculator that defines the right-hand-side ODE function
    PlantXCalc = PlantX.calculate
    
    input_data = [
        ODEModelSolver,
        PlantX,
        ForcingDataX.time_axis,
        ForcingDataX.inputs,
        ForcingDataX.reset_days,
        ForcingDataX.zero_crossing_indices,
        ForcingDataX.time_nday_f,
        ForcingDataX.time_doy_f,
        ForcingDataX.time_year_f
    ]
    return input_data


