# The Data

Data for this project comes from the *[DWD - Deutscher Wetterdienst](https://github.com/earthobservations/wetterdienst)*. The API supports access to multiple weather services (e.g. Mosmix - statistical optimized scalar forecasts extracted from weather model) as well as other related agencies (e.g. NOAA - National Oceanic And Atmospheric Administration). It provides the relevant data for this project in form of historical weather observations. Coming from measuring stations across Germany, time series data ranges from minute-by-minute to yearly frequency dating back from the present up to the last ~300 years. However, weather data seems to be heterogenous to some extent: Not all weather stations provide the same data (i.e. variables, quality, etc.) across the same time span.


Station-ID 1766 -> Muenster

| Variable | Description                                                            | Range | Type |
|----------|------------------------------------------------------------------------|----------|------|
| TEMPERATURE_AIR_MEAN_200 | text                                                   | [-20,40] | int  |
| WIND_SPEED               | text                                                   | [0,14]   | int  |
| CLOUD_COVER_TOTAL        | In meteorology, an okta is a unit of measurement used 
                             to describe the amount of cloud cover at any given 
                             location. Sky condititions are estimated in terms of 
                             how many eighths of the sky are covered in cloud, 
                             ranging from 0 oktas (completely clear sky) to 8 oktas
                             (completely overcast). Apparently, there exists an 
                             extra cloud cover indicator '9' meaning  that the sky 
                             is totally obscured (i.e  hidden from view), usually
                             due to dense fog or heavy snow.                          | [0,8]    | int  |


A full list of available variables can be found *[here](https://wetterdienst.readthedocs.io/en/latest/data/parameters.html)*



Note: variable names will be altered during the process of data preprocessing.

Sources:
*[Wetterdienst - Docs](https://wetterdienst.readthedocs.io/en/latest/index.html)*
*[Wetterdienst - Overview](https://www.dwd.de/EN/ourservices/cdc/cdc_ueberblick-klimadaten_en.html)*


