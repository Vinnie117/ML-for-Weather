import pandas as pd

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


# build differences (e.g. change in temperature ~ velocity of variable)

# build differences in differences (e.g. change in temperature change ~ acceleration of variable)

print(df.dtypes)
print(df.shape)

# Save processed data
df.to_csv(r'A:\Projects\ML-for-Weather\data\processed\df.csv', header=True, index=False)