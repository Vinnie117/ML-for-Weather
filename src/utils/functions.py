import logging
from sympy import false
import sys
sys.path.append('A:\Projects\ML-for-Weather\src') 
from wetterdienst.provider.dwd.observation import DwdObservationRequest
from wetterdienst import Settings
from config import data_config
import os


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






