Using Wetterdienst Explorer 
- In powershell terminal:
    - pip install wetterdienst[explorer]
    - wetterdienst explorer
    - open in browser http://localhost:7891

Notes:
- Do not use daily data for e.g. temperature. It is not informative as the mean temperature for the whole day is calculated
- Prefer hourly data.
    - from "1990-01-01" to "2020-01-01" gives ca. 260,000 data points
- MOSMIX data (network = "MOSMIX) does not provide historical data but future predictions


Notes on Data 
- Parameter availability depends on resolution
    - parameter=["TEMPERATURE_AIR"]
        - resolution = "hourly"
    - in df: column "dataset" only has values "temperature_air"
    - in df: column "parameter" has two different values "humidity" and "temperature_mean_200"
        - 200 is for the height (2m) where temperature is measured
    - But NAs for Muenster (station_id = 3404) -> use station_id = 1766 (Muenster/Osnabrueck)!
        - has all values
    - parameter=["PRECIPITATION_MORE"]
        - resolution = ["daily"]


Tests:
- parameter = ["TEMPERATURE_AIR"] + df2.loc[df2["parameter"] == 'temperature_air_mean_200'] \
vs.
- parameter = ["TEMPERATURE_AIR_MEAN_200"]
- -> sorted by highest values -> same data

- You can indicate multiple variables
    - parameter = ["TEMPERATURE_AIR", "HUMIDITY"]

- Get ALL historical data
    - period=Period.HISTORICAL    

