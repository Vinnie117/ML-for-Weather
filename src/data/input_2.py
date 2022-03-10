from wetterdienst import Wetterdienst, Resolution, Period
from wetterdienst.provider.dwd.observation import DwdObservationDataset
from wetterdienst.provider.dwd.observation import DwdObservationRequest
import pandas as pd

pd.options.display.max_columns = 8

API = Wetterdienst(provider="dwd", network="observation")

#### Explore parameters
observations_meta = API.discover(filter_=Resolution.DAILY)

# available parameter sets
#print(observations_meta)

# Available individual parameters
observations_meta_flat = API.discover(filter_=Resolution.DAILY, flatten=False)
#print(observations_meta_flat)
####

#### Explore available cities for one parameter

# Parameter availability depends on resolution:
# https://wetterdienst.readthedocs.io/en/latest/data/coverage/dwd.html#overview
# https://github.com/earthobservations/wetterdienst/blob/106a2fa9f887983281a6886c15bb3a845850dfb7/wetterdienst/provider/dwd/observation/metadata/dataset.py#L21

stations = DwdObservationRequest(parameter=DwdObservationDataset.TEMPERATURE_AIR,
                                 resolution=Resolution.HOURLY,
                                 period=Period.HISTORICAL)

print(stations.all().df.head())
####

#### Get data for one specific weather station

# https://github.com/earthobservations/wetterdienst
request = DwdObservationRequest(parameter=["TEMPERATURE_AIR"],
                                resolution="hourly",
                                start_date="1990-01-01",  # if not given timezone defaulted to UTC
                                end_date="2020-01-01",  # if not given timezone defaulted to UTC
                                ).filter_by_station_id(station_id=(1766))

# the specified weather station
df1 = request.df.head()
print(df1)
# the data
df2 = request.values.all().df
print(df2)

'''
Im Data Viewer

parameter=["TEMPERATURE_AIR"]
- Spalte "dataset" hat nur Werte "temperature air"
- Spalte "parameter" hat 2 verschiedene Werte "humidity" und "temperature_mean_200" 
- aber NaN-Werte für Münster (3404) -> für 1766 gibt es alle Werte!!


parameter=["PRECIPITATION_MORE"]

'''



# https://wetterdienst.readthedocs.io/en/latest/usage/python-examples.html
# -> Alle stations laden und dann nach stations id filtern?

stations2 = API(parameter=DwdObservationDataset.PRECIPITATION_MORE,
               resolution=Resolution.DAILY,
               period=Period.HISTORICAL)

print('\n', next(stations2.all().values.query()))










