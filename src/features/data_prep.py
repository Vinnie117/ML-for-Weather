import pandas as pd

# load data
df = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 
df = df.drop(columns=['Unnamed: 0', "station_id", "dataset"])
print(df.dtypes)

df['date'] = pd.to_datetime(df['date'])
print(df.dtypes)

pivoted = df.pivot(index="date", columns="parameter", values="value")


# pivoted table -> but date is empty -> why?
# -> only empty in Data Viewer? 
# -> yes, Data Viewer is buggy? Csv can be written and is correct

pivoted.to_csv(r'A:\Projects\ML-for-Weather\data\raw\pivoted.csv', header=True)





# Try this:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html
print("END")