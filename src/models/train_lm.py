import pandas as pd


# load data
df_lm = pd.read_csv('A:\Projects\ML-for-Weather\data\processed\df.csv') 

print(df_lm.dtypes)

# Start with simple naive model: explain temperature at t with temperature at t-1


print("END")