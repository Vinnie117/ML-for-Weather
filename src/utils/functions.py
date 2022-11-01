import logging
from sympy import false
import sys
sys.path.append('A:\Projects\ML-for-Weather\src') 
from wetterdienst.provider.dwd.observation import DwdObservationRequest
from wetterdienst import Settings
from config import data_config
import os
import pandas as pd

def download(cfg: data_config):
    ''''
    This function downloads raw data from DWD API
    '''

    logging.info("DOWNLOADING DATA FROM DWD API")

    # Changing temperature units to Celsius
    Settings.si_units = false

    historical_data = DwdObservationRequest(
        parameter=["TEMPERATURE_AIR_MEAN_200","WIND_SPEED", "CLOUD_COVER_TOTAL"],
        resolution=cfg.weather.resolution,
        start_date=cfg.weather.start_date,  # if not given timezone defaulted to UTC
        end_date=cfg.weather.end_date,
    ).filter_by_station_id(station_id=cfg.weather.station)

    df = historical_data.values.all().df

    path = os.path.join(os.getcwd(), 'data_dvc', 'raw', 'training')
    name = 'data_' + str(cfg.weather.station) + '_from_' + cfg.weather.start_date + '_to_' + cfg.weather.end_date
    format = 'csv'
    file = os.path.join(path, name + '.' + format)

    df.to_csv(file, header=True, index=False)


def data_loader(data, cfg: data_config):

    use = list(cfg['data'].keys())[list(cfg['data'].values()).index(cfg['data'][data])]   
    logging.info('LOAD DATA FOR {use} FROM DIRECTORY'.format(use = use.upper()))

    # load data (make this better! function arg should directly reference cfg)
    try:
        df_raw = pd.read_csv(cfg['data'][data])
    except:
        print("Function argument type must be of available type in config -> data")

    # clean up and prepare
    data = df_raw.drop(columns=['station_id', 'dataset'])
    data = data.pivot(index='date', columns='parameter', values='value').reset_index()
    
    # renaming
    for i in cfg.vars_old:
        data = data.rename(columns={cfg.vars_old[i]: cfg.vars_new[i]})
    
    # ordering
    data.insert(1, cfg.vars_new.temp, data.pop(cfg.vars_new.temp))

    return data




