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
