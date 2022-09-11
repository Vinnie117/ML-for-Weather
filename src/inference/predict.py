from joblib import dump, load
import pandas as pd

# load model
model_joblib = load(r'A:\Projects\ML-for-Weather\artifacts\models\xgb.joblib') 

#load test data
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv', delimiter=',', header=0)
X_test = test.iloc[:, 1:]

# single data point for inference
inf_point = X_test.iloc[:1]

print(model_joblib)
print(X_test)
print(inf_point)


preds = model_joblib.predict(X_test)
print(preds)

inf_pred = model_joblib.predict(inf_point)
print(inf_pred)










