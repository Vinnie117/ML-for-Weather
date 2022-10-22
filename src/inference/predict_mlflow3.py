import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from config import data_config
import mlflow
import pandas as pd
from features.pipeline_dataprep import pd_df, cfg
from sklearn.pipeline import Pipeline
from inference_classes import IncrementTime, SplitTimestamp, IncrementLaggedAccelerations
from inference_classes import IncrementLaggedUnderlyings, IncrementLaggedVelocities
from wetterdienst.provider.dwd.observation import DwdObservationRequest
from wetterdienst import Settings
from wetterdienst import Wetterdienst
from sympy import false
from time import time
from wetterdienst import Resolution, Period
from wetterdienst.provider.dwd.observation import DwdObservationDataset


# manual look-up and copying -> automating possible?
model_temperature = 'runs:/c8e2ca3172b64f1999116b4a8b290e7e/best_estimator'
model_cloud_cover = 'runs:/144c1be3dab346a19c95605c46c675f9/best_estimator'
model_wind_speed = 'runs:/fa8aed07ece5402f923ea24776d5b405/best_estimator'

# Load all models as a PyFuncModel.
model_temperature = mlflow.pyfunc.load_model(model_temperature)
model_cloud_cover = mlflow.pyfunc.load_model(model_cloud_cover)
model_wind_speed = mlflow.pyfunc.load_model(model_wind_speed)

#################################################################################
#### Shorten inference
Settings.si_units = false
API = Wetterdienst(provider="dwd", network="observation")


#print(pd_df.tail())

inference_data = DwdObservationRequest(parameter=["TEMPERATURE_AIR_MEAN_200","WIND_SPEED", "CLOUD_COVER_TOTAL"],
                            resolution="hourly",
                            start_date='2019-12-31',
                            end_date='2020-01-01',
                            ).filter_by_station_id(station_id=(1766))


'''
Converting the request to dataframe is VERY slow
Why does it take so long? around 30 seconds!

- reason is the .all() method
- inference_data.values is of type DwdObservationValues
- run cron job to get latest available data for inference in cloud?

For now: save pseudo inference data locally
'''


t0 = time()
inference_data = inference_data.values.all().df
duration = time() - t0
print(type(inference_data))
print(inference_data)
print(duration)

inference_data.to_csv(r'A:\Projects\ML-for-Weather\data_dvc\raw\pseudo_inference.csv', header=True, index=False)




#################################################################################

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
        ("increment lagged velos", IncrementLaggedVelocities()),
        ("increment lagged accs", IncrementLaggedAccelerations())
        ])

    return pipe



def walking_inference(walking_df, end_date):
    '''
    Function for incremental inference (row by row)
    '''

    while walking_df['timestamp'].iloc[-1] < pd.Timestamp(end_date):

        # get newest point of dataframe
        latest = walking_df.iloc[-1:]

        # predict on latest row of dataframe
        pred_temperature = model_temperature.predict(pd.DataFrame(latest))[0]
        pred_cloud_cover = model_cloud_cover.predict(pd.DataFrame(latest))[0]
        pred_wind_speed = model_wind_speed.predict(pd.DataFrame(latest))[0]

        # append predictions on dataframe
        walking_df = pd.concat([walking_df, pd.DataFrame.from_records([{
            'temperature':pred_temperature, 
            'cloud_cover':pred_cloud_cover, 
            'wind_speed':pred_wind_speed}])])

        # Apply pipeline (inference_prep) on dataframe
        walking_df = pipeline_inference_prep(cfg = cfg).fit_transform(walking_df)

        # exit while loop once end date of incremental inference is reached
        if walking_df['timestamp'].iloc[-1] == end_date:
            break

    return walking_df

df_walking_inference = walking_inference(pd_df, "2020-01-01 03:00:00")
print(df_walking_inference[['year', 'month', 'day', 'hour', 'temperature', 'temperature_lag_1',
                            'temperature_velo_1_lag_1', 'temperature_velo_1_lag_2',
                            'temperature_velo_2_lag_1','temperature_velo_2_lag_3','temperature_acc_1_lag_3']].tail(10))

print("END")