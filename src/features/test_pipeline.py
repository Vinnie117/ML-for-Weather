from posixpath import split
import pandas as pd
from sklearn.pipeline import Pipeline
from pipeline_classes import InsertLags
#from pipeline_classes import df
from pipeline_classes import Debugger
from pipeline_classes import Cleaner


# load data
df_raw = pd.read_csv(r'A:\Projects\ML-for-Weather\data\raw\test_simple.csv') 

pipe = Pipeline([
    ("debug1", Debugger()),
    ('clean', Cleaner()),
    ("debug2", Debugger()),
    ("lags", InsertLags([1,2,3,24])),
    ("debug3", Debugger())
])

# Pipeline creates lags and prints the data
data = pipe.fit_transform(df_raw) 



data.to_csv(r'A:\Projects\ML-for-Weather\data\processed\df.csv', header=True, index=False)
print("END")