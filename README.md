# ML-for-Weather
This is a playground project for ML models to explore weather prediction. There are interesting similarities with as well as distinctions to financial markets when applying ML to weather:
- Both predominantly work with time series data, however, weather data seems to be stationary most of the time.
- In the short term, weather appears to be very well predictable (> 90% adj. R2!), whereas achieving more than 50% predictive accuracy in finance would yield tremendous wealth :P
- Predicting weather is not a competition or has no self-fulfilling prophecies. In financial markets, prediction models lose value over time, e.g. due to changing environments or market participants exploiting mispricings which then disappear.
- When setting up forecasting models / trading strategies, there is no need to consider transactions costs, such as trading fees or taxes.


Currently, the project architecture is as follows:
![excalidraw new(2)](https://user-images.githubusercontent.com/52510339/200659666-da297f72-d25f-404b-a002-ba2642deef80.png)




## Usage
The central entrypoint to use this project is "src\conf\config.yaml" where parameters for feature engineering and model training can be set.
More documenation is here: https://github.com/Vinnie117/ML-for-Weather/tree/main/Docs
