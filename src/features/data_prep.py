import pandas as pd

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

# prepare and clean data
df = df_raw.drop(columns=['Unnamed: 0', "station_id", "dataset"])
df['date'] = pd.to_datetime(df['date'])

df = df.pivot(index="date", columns="parameter", values="value").reset_index()

# Save processed data
df.to_csv(r'A:\Projects\ML-for-Weather\data\interim\df.csv', header=True)