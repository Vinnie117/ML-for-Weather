import pandas as pd

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

# prepare and clean data
df = df_raw.drop(columns=["station_id", "dataset"])


df = df.pivot(index="date", columns="parameter", values="value").reset_index()

#####################################################################
# get time (UTC) and convert to local (UTC +1 hour = CET)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

df['date_cet'] = pd.to_datetime(df['date']).dt.tz_convert('Europe/Berlin')

# remove timezone
df['test'] = pd.to_datetime(df['date']).dt.tz_localize(None)

# convert to CET, then remove tz
df['test2'] = pd.to_datetime(df['date']).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)

df["test3"]= pd.to_datetime(df["test2"])


# extract informative variables
#df['month'] =  df['date'].dt.month
#df['day'] =  df['date'].dt.day
#df['hour'] =  df['date'].dt.hour







print(df.dtypes)

# Save processed data
df.to_csv(r'A:\Projects\ML-for-Weather\data\processed\df.csv', header=True, index=False)