import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import mlflow
import pandas as pd
from features.pipeline_dataprep import pd_df
from sklearn.pipeline import Pipeline
from inference_classes import IncrementTime, SplitTimestamp

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
print(test.iloc[-10:,0:12])


#############################################################################
######## transform data for next (row of) inference

# # increment timestamp
# test.loc[test.index[-1], "timestamp"] = test.loc[test.index[-2], "timestamp"]+ pd.to_timedelta(1,unit='h')
# print(test.iloc[-10:,0:12])

# # Split timestamp to year, month, day, hour
# test.loc[test.index[-1], "hour"]  =  test.loc[test.index[-1], "timestamp"].hour
# test.loc[test.index[-1], "day"]  =  test.loc[test.index[-1], "timestamp"].day
# test.loc[test.index[-1], "month"]  =  test.loc[test.index[-1], "timestamp"].month
# test.loc[test.index[-1], "year"]  =  test.loc[test.index[-1], "timestamp"].year
# print(test.iloc[-10:,0:12])

# collect steps in pipeline
walking_inference_dataprep = Pipeline([
    ("increment time", IncrementTime()), 
    ("split timestamp", SplitTimestamp())
    ])

test = walking_inference_dataprep.fit_transform(test)

print(test.iloc[-10:,0:12])

# add pipeline step for lags

print("END")