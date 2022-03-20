from sklearn.pipeline import Pipeline
from pipeline_classes import InsertLags
from pipeline_classes import df
from pipeline_classes import Debugger

pipe = Pipeline([
    ("lags", InsertLags([1,2,3,24])),
    ("debug", Debugger())
])

# Pipeline creates lags and prints the data
data = pipe.fit_transform(df)

data.to_csv(r'A:\Projects\ML-for-Weather\data\processed\df.csv', header=True, index=False)
print("END")