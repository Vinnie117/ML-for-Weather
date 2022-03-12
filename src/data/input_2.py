from numpy import dtype
from sympy import false
from wetterdienst import Wetterdienst, Resolution, Period
from wetterdienst.provider.dwd.observation import DwdObservationDataset
from wetterdienst.provider.dwd.observation import DwdObservationRequest
import pandas as pd
from wetterdienst import Settings

# Changing temperature units to Celsius
Settings.si_units = false

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

# ALl historical data
hist = DwdObservationRequest(parameter=["TEMPERATURE_AIR_MEAN_200"],
                            resolution="hourly",
                            period=Period.HISTORICAL,
                            ).filter_by_station_id(station_id=(1766))


df_hist = hist.values.all().df


# https://github.com/earthobservations/wetterdienst
request = DwdObservationRequest(parameter=["TEMPERATURE_AIR_MEAN_200"],
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
print(type(df2))

#df3 = df2.loc[df2['parameter'] == 'temperature_air_mean_200']