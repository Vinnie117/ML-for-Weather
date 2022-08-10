import xgboost as xgb
import pandas as pd

# get data
train = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', delimiter=',', header=0)
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', delimiter=',', header=0)
X_train = train
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]