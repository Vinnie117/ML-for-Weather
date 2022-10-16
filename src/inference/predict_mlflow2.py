import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from config import data_config
import mlflow
import pandas as pd
from features.pipeline_dataprep import pd_df, cfg
from sklearn.pipeline import Pipeline
from inference_classes import IncrementTime, SplitTimestamp
from inference_classes import IncrementLaggedUnderlyings, IncrementLaggedVelocities

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
print(latest.dtypes)

# Predict on a Pandas DataFrame.
pred_temperature = model_temperature.predict(pd.DataFrame(latest))[0]
pred_cloud_cover = model_cloud_cover.predict(pd.DataFrame(latest))[0]
pred_wind_speed = model_wind_speed.predict(pd.DataFrame(latest))[0]



# append data
test = pd_df.append({'temperature':pred_temperature, 
                    'cloud_cover':pred_cloud_cover, 
                    'wind_speed':pred_wind_speed}, ignore_index=True)
print(test.iloc[-10:,0:12])


#############################################################################
######## transform data for next (row of) inference

'''
Reason for walking inference (add this to documentation later): Inference in a distant point in time
requires all features of the previous row to be present! So we predict the row for row from the last
known row, i.e. the end of the test data

'''

# collect steps in pipeline
def pipeline_inference_prep(cfg: data_config):

    pipe = Pipeline([
        ("increment time", IncrementTime()), 
        ("split timestamp", SplitTimestamp()),
        ("increment lagged underlyings", IncrementLaggedUnderlyings(vars = cfg.transform.vars, lags = cfg.diff.lags)),
        ("increment lagged velos", IncrementLaggedVelocities())
        ])

    return pipe

# Apply pipeline (inference_prep) on dataframe
df_walking_inference = pipeline_inference_prep(cfg = cfg).fit_transform(test)

# Check if new row is appended
print(df_walking_inference[['year', 'month', 'day', 'hour', 'temperature', 'temperature_lag_1',
                  'temperature_velo_1_lag_1', 'temperature_velo_1_lag_2',
                  'temperature_velo_2_lag_1','temperature_velo_2_lag_3']].tail(10))


#### START of inference procedure
# get newest point of dataframe
latest = df_walking_inference.iloc[-1:]

# predict on latest row of dataframe
pred_temperature = model_temperature.predict(pd.DataFrame(latest))[0]
pred_cloud_cover = model_cloud_cover.predict(pd.DataFrame(latest))[0]
pred_wind_speed = model_wind_speed.predict(pd.DataFrame(latest))[0]

# append predictions on dataframe
df_walking_inference2 = df_walking_inference.append({'temperature':pred_temperature, 
                                                     'cloud_cover':pred_cloud_cover, 
                                                     'wind_speed':pred_wind_speed}, ignore_index=True)
print(df_walking_inference2.iloc[-10:,0:12])

# Apply pipeline (inference_prep) on dataframe
df_walking_inference2 = pipeline_inference_prep(cfg = cfg).fit_transform(df_walking_inference2)

# Check if new row is appended
print(df_walking_inference2[['year', 'month', 'day', 'hour', 'temperature', 'temperature_lag_1',
                  'temperature_velo_1_lag_1', 'temperature_velo_1_lag_2',
                  'temperature_velo_2_lag_1','temperature_velo_2_lag_3','temperature_acc_1_lag_3']].tail(10))

# get newest point of dataframe
latest = df_walking_inference2.iloc[-1:]


# ...





# These cols still need to be incremented
na = test.columns[test.isna().any()].tolist()
print(na)

print("END")