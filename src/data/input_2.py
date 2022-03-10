from wetterdienst import Wetterdienst, Resolution, Period
from wetterdienst.provider.dwd.observation import DwdObservationDataset
from wetterdienst.provider.dwd.observation import DwdObservationRequest

API = Wetterdienst("dwd", "observation")

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








