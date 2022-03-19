import pandas as pd
from sklearn.model_selection import train_test_split

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

# prepare and clean data
df = df_raw.drop(columns=["station_id", "dataset"])

df = df.pivot(index="date", columns="parameter", values="value").reset_index()

# convert to CET (UTC +1), then remove tz
df['timestamp'] = pd.to_datetime(df['date']).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
df = df.drop('date', 1)

# extract informative variables
df['month'] =  df['timestamp'].dt.month
df['day'] =  df['timestamp'].dt.day # is this informative? Does it connect to "month"?
df['hour'] =  df['timestamp'].dt.hour

# build lags
df['temperature_air_mean_200_lag_1'] = df['temperature_air_mean_200'].shift(1)
df['temperature_air_mean_200_lag_2'] = df['temperature_air_mean_200'].shift(2)
df['cloud_cover_total_lag_1'] = df['cloud_cover_total'].shift(1)
df['cloud_cover_total_lag_2'] = df['cloud_cover_total'].shift(2)
df['wind_speed_lag_1'] = df['wind_speed'].shift(1)
df['wind_speed_lag_2'] = df['wind_speed'].shift(2)

# build differences (e.g. change in temperature ~ velocity of variable)

# build differences in differences (e.g. change in temperature change ~ acceleration of variable)

# build variable of previous day (With 50%, tomorrow's weather will be the same as today's weather)


# Naming

# Ordering
df = df[['timestamp', 'month', 'day', 'hour', 
'temperature_air_mean_200', 'temperature_air_mean_200_lag_1', 'temperature_air_mean_200_lag_2',
'cloud_cover_total', 'cloud_cover_total_lag_1', 'cloud_cover_total_lag_2', 
'wind_speed', 'wind_speed_lag_1', 'wind_speed_lag_2']]

print(df.dtypes)
print(df.shape)

# Save processed data
df.to_csv(r'A:\Projects\ML-for-Weather\data\processed\df.csv', header=True, index=False)