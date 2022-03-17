from matplotlib.pyplot import hist
import pandas as pd
from pytz import timezone
from sympy import false
from wetterdienst import Wetterdienst, Resolution, Period
from wetterdienst.provider.dwd.observation import DwdObservationDataset
from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationPeriod
from wetterdienst import Settings

# Changing temperature units to Celsius
Settings.si_units = false

API = Wetterdienst(provider="dwd", network="observation")



# Test call
# avoid timezone issue with period=DwdObservationPeriod.HISTORICAL?
# -> is the same as specifying "start_date=..." and "end_date=..." -> so UTC default here as well(?)

historical_data = DwdObservationRequest(parameter=["TEMPERATURE_AIR_MEAN_200","WIND_SPEED", "CLOUD_COVER_TOTAL"],
                            resolution="hourly",
                            start_date="2019-01-01",  # if not given timezone defaulted to UTC
                            end_date="2020-01-01",
                            ).filter_by_station_id(station_id=(1766))


'''
# Historical data: do all variables have the same starting point, possibly shortening the time window of available data?
historical_data = DwdObservationRequest(parameter=["TEMPERATURE_AIR_MEAN_200", "HUMIDITY", "CLOUD_COVER_TOTAL", "PRECIPITATION_HEIGHT", "WIND_SPEED"],
                            resolution="hourly",
                            period=DwdObservationPeriod.HISTORICAL,                            
                            ).filter_by_station_id(station_id=(1766))
'''


df_hist = historical_data.values.all().df

df_hist.to_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv', header=True, index=False)