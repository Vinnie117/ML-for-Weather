import mlflow
import pandas as pd
logged_model = 'runs:/bc568713f8b24e949df646ec2c054505/best_estimator'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

#load test data
test = pd.read_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv', delimiter=',', header=0)
X_test = test.iloc[:, 1:]
# single data point for inference
inf_point = X_test.iloc[:1]

# Predict on a Pandas DataFrame.
import pandas as pd
inference = loaded_model.predict(pd.DataFrame(inf_point))

print(inference)

