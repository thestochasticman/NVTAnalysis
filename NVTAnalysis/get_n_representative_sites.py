from PaddockTS.Data.environmental import download_environmental_data
from PaddockTS.query import Query
from pandas import read_csv
from os.path import exists
from datetime import date

def get_n_representative_sites(tmp_dir: str, out_dir: str):
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
            end_time=date.fromisoformat(row['HarvestDate']),
            out_dir=out_dir,
            tmp_dir=tmp_dir,
            stub=row['TrialCode']
        )
        queries += [query]
        if not exists(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv'):
            download_environmental_data(query)

        df = read_csv(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv')
        
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})
            df.to_csv(f'{query.stub_tmp_dir}/environmental/{query.stub}_DAESim_forcing.csv', index='Date')
    return queries