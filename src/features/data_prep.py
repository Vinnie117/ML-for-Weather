import pandas as pd

# load data
df = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 
df = df.drop(columns=['Unnamed: 0', "station_id", "dataset"])
print(df.dtypes)

df['date'] = df['date'].astype(str)
print(df.dtypes)

pivoted = df.pivot(index="date", columns="parameter", values="value")

pivoted["date"] = pivoted.index

# pivoted table -> but date is empty -> why?
# -> only empty in Data Viewer? 
# -> Date column is transformed to an index -> needs to be a separate column!!!
# or use df = df.reset_indes(level=0)
# google: pandas unstack one column
# what is type of date column?


pivoted.to_csv(r'A:\Projects\ML-for-Weather\data\raw\pivoted.csv', header=True)





# Try this:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html
print("END")