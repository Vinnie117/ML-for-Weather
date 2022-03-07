from wetterdienst import Wetterdienst, Resolution, Period
from wetterdienst.provider.dwd.observation import DwdObservationDataset
from wetterdienst.provider.dwd.observation import DwdObservationRequest

API = Wetterdienst("dwd", "observation")

observations_meta = API.discover(filter_=Resolution.DAILY)

print(API)
print("#################")

# Available parameter sets
print(observations_meta)
print("#################")

observations_meta_flat = API.discover(filter_=Resolution.DAILY, flatten=False)

print(observations_meta_flat)
print("#################")


stations = DwdObservationRequest(parameter=DwdObservationDataset.PRECIPITATION_MORE,
                                 resolution=Resolution.DAILY,
                                 period=Period.HISTORICAL)
print(stations)
print("#################")


# Stations ID of Münster is 03404
print(stations.all().df.head())
print("#################")


# Use this! 
# https://github.com/earthobservations/wetterdienst















