import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import mlflow
import pandas as pd
from features.pipeline_dataprep import pd_df

# manual look-up and copying -> automating possible?
model_temperature = 'runs:/c8e2ca3172b64f1999116b4a8b290e7e/best_estimator'
model_cloud_cover = 'runs:/144c1be3dab346a19c95605c46c675f9/best_estimator'
model_wind_speed = 'runs:/fa8aed07ece5402f923ea24776d5b405/best_estimator'

# Load all models as a PyFuncModel.
model_temperature = mlflow.pyfunc.load_model(model_temperature)
model_cloud_cover = mlflow.pyfunc.load_model(model_cloud_cover)
model_wind_speed = mlflow.pyfunc.load_model(model_wind_speed)

# single and latest data point for inference
latest = pd_df.iloc[-1:]
print(latest)


# Predict on a Pandas DataFrame.
pred_temperature = model_temperature.predict(pd.DataFrame(latest))[0]
pred_cloud_cover = model_cloud_cover.predict(pd.DataFrame(latest))[0]
pred_wind_speed = model_wind_speed.predict(pd.DataFrame(latest))[0]

print(pred_temperature)
print(pred_cloud_cover)
print(pred_wind_speed)


# append data
test = pd_df.append({'temperature':pred_temperature, 
                    'cloud_cover':pred_cloud_cover, 
                    'wind_speed':pred_wind_speed}, ignore_index=True)
print(test.tail(5))


# Not all underlyings are in pd_df -> adjust pd_df in pipline_data_prep_classes to contain all?

print("END")