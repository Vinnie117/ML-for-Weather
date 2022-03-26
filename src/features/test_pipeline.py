import pandas as pd
from sklearn.pipeline import Pipeline
from pipeline_classes import InsertLags
from pipeline_classes import Debugger
from sklearn.model_selection import train_test_split
from pipeline_classes import clean

# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 
df = clean(df_raw)



# Splitting here to prevent information leakage
train, test = train_test_split(df, test_size=0.2, shuffle = False)


pipe = Pipeline([
    ("lags", InsertLags([1,2,3,24])),
    ("debug3", Debugger())
])



# Pipeline creates lags and prints the data
data = pipe.fit_transform(test) 




data.to_csv(r'A:\Projects\ML-for-Weather\data\processed\df.csv', header=True, index=False)
print("END")