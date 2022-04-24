# ML-for-Weather
This is a playground project for ML models to explore weather prediction. There are interesting similarities with as well as distinctions to financial markets when applying ML to weather:
- Both predominantly work with time series data, however, weather data seems to be stationary most of the time.
- In the short term, weather appears to be very well predictable (> 90% adj. R2!), whereas achieving more than 50% predictive accuracy in finance would yield tremendous wealth :P
- Predicting weather is not a competition or has no self-fulfilling prophecies. In financial markets, prediction models lose value over time, e.g. due to changing environments or market participants exploiting mispricings which then disappear.
- When setting up forecasting models, there is no need to consider transactions costs, such as trading fees or taxes.


Currently, the project architecture is as follows:
![grafik](https://user-images.githubusercontent.com/52510339/164978858-adae8d85-c4c4-4316-be28-4d6daa9f62cf.png)
