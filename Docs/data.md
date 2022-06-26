# The Data

Data for this project comes from the *[DWD - Deutscher Wetterdienst](https://github.com/earthobservations/wetterdienst)*. The API supports access to multiple weather services (e.g. Mosmix - statistical optimized scalar forecasts extracted from weather model) as well as other related agencies (e.g. NOAA - National Oceanic And Atmospheric Administration). It provides the relevant data for this project in form of historical weather observations. Coming from measuring stations across Germany, time series data ranges from minute-by-minute to yearly frequency dating back from the present up to the last ~300 years. However, weather data seems to be heterogenous to some extent: Not all weather stations provide the same data (i.e. variables, quality, etc.) across the same time span.

For testing purposes, only data from one weather station (ID 1766) has been explored so far. As the project is work-in-progress, the list of data will be updated on the go. Currently, the following variables have been 
requested already:

<!---
TABLE for variable description
- made and copied with https://www.tablesgenerator.com/markdown_tables#     
    -->
| Variable | Description | Range | Type |
|---|---|---|---|
| TEMPERATURE_AIR_MEAN_200 | The mean temperature at 2 meters above ground in the select frequency in °C | [-20,40] | int |
| WIND_SPEED | Measured in m/s | [0,14] | int |
| CLOUD_COVER_TOTAL | In meteorology, an okta is a unit of measurement used to describe the amount of cloud cover at any given location. Sky condititions are estimated in terms of how many eighths of the sky are covered in cloud, ranging from 0 oktas (completely clear sky) to 8 oktas (completely overcast). Apparently, there exists an extra cloud cover indicator '9' meaning that the sky is totally obscured (i.e  hidden from view), usually due to dense fog or heavy snow. | [0,8] | int |

A full list of available variables can be found *[here](https://wetterdienst.readthedocs.io/en/latest/data/parameters.html)*. Note that downloaded variable names differ from variable names for model training as they
will be altered during *[data preprocessing](https://github.com/Vinnie117/ML-for-Weather/blob/main/Docs/preprocessing.md)* 

When using the API, data comes out in this form:
![grafik](https://user-images.githubusercontent.com/52510339/175810749-8c117922-5f34-4c24-943a-e82e794f6311.png)




## Sources:
- *[Wetterdienst - Docs](https://wetterdienst.readthedocs.io/en/latest/index.html)*
- *[Wetterdienst - Overview](https://www.dwd.de/EN/ourservices/cdc/cdc_ueberblick-klimadaten_en.html)*

