from config import data_config
from sklearn.pipeline import Pipeline
import pandas as pd
from inference_classes import IncrementTime, SplitTimestamp, IncrementLaggedAccelerations
from inference_classes import IncrementLaggedUnderlyings, IncrementLaggedVelocities
from inference.pipeline_inference_features_classes import Times, Velocity, Acceleration, InsertLags, Scaler, Prepare


def pipeline_features_inference(cfg: data_config):
    '''
    Pipeline to prepare downloaded data (AFTER data_loader()) for inference pipeline
    '''

    pipe = Pipeline([

        ("times", Times()),
        ('velocity', Velocity(vars=cfg.transform.vars, diff=cfg.diff.diff)),   
        ('acceleration', Acceleration(vars=cfg.transform.vars, diff=cfg.diff.diff)),  # diff of 1 row between 2 velos
        ('lags', InsertLags(diff=cfg.diff.lags)),

        # Standardization works fine but exclude for now
        # ('scale', Scaler(target = cfg.model.target, std_target=False)),
         
        ('cleanup', Prepare(target = cfg.model.target, predictors=cfg.model.predictors, vars = cfg.transform.vars))
        ])
        

    return pipe


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



def walking_inference(cfg: data_config, walking_df, end_date, model_temperature, model_cloud_cover, model_wind_speed):
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
        walking_df = pipeline_inference_prep(cfg=cfg).fit_transform(walking_df)

        # exit while loop once end date of incremental inference is reached
        if walking_df['timestamp'].iloc[-1] == end_date:
            break

    return walking_df