from config import data_config
from sklearn.pipeline import Pipeline
import pandas as pd
from src.inference.inference_classes import IncrementTime, SplitTimestamp, IncrementLaggedAccelerations
from src.inference.inference_classes import IncrementLaggedUnderlyings, IncrementLaggedVelocities
from inference.pipeline_inference_features_classes import Times, Velocity, Acceleration, InsertLags, Scaler, Prepare
import mlflow
from tqdm import tqdm
import numpy as np
import logging

def pipeline_features_inference(cfg: data_config):
    '''
    Pipeline to prepare downloaded data (AFTER data_loader()) for inference pipeline
    '''

    logging.info('PREPARE DATA FOR INFERENCE')

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


def pipeline_inference_prep(cfg: data_config):
    '''
    Used by walking_inference()
    '''

    pipe = Pipeline([
        ("increment time", IncrementTime()), 
        ("split timestamp", SplitTimestamp()),
        ("increment lagged underlyings", IncrementLaggedUnderlyings(vars = cfg.transform.vars, lags = cfg.diff.lags)),
        ("increment lagged velos", IncrementLaggedVelocities()),
        ("increment lagged accs", IncrementLaggedAccelerations())
        ])

    return pipe


def model_loader():
    '''
    This function automatically returns the best models (run from e.g. GridSearchCV) in a dict
    '''

    logging.info('FETCHING MODELS FROM MLFLOW DIRECTORY')

    # search mlflow experiments by tag runName
    df = mlflow.search_runs(['3'], filter_string="tags.mlflow.runName ILIKE '%XGB, target:%'")

    # sort by adjusted_r2,  then take  first element ( = minimum) in each runName group:
    df = df.sort_values("metrics.adjusted_r2").groupby("tags.mlflow.runName", as_index=False).first()

    # now load all different models into a dict
    models = {}
    for i, j in zip(df['run_id'], df['tags.mlflow.runName']):
        
        # construct model_name
        var = j.split('target: ')[1]
        model_name = "model_" + var 

        # load and assign all models as a PyFuncModel
        models[model_name] = mlflow.pyfunc.load_model('runs:/' + i + '/best_estimator')

    logging.info('THE MODEL IDs USED ARE: \n {models}'.format(models = models))
    return models



def walking_inference(cfg: data_config, walking_df, end_date):
    '''
    Function for incremental inference (row by row)
    '''
    
    models = model_loader()
    predictions = {}

    # calculate inference period in hours (for progress bar)
    diff = pd.Timestamp(end_date) - walking_df['timestamp'].iloc[-1]
    hours = (diff / np.timedelta64(1, 'h')) - 1
    pbar = tqdm(total = 100 )

    while walking_df['timestamp'].iloc[-1] < pd.Timestamp(end_date):

        # get newest point of dataframe (i.e. latest complete row)
        latest = walking_df.iloc[-1:]

        # get models and collect predictions in dict
        for i in models:

            pred_name = i.split('model_')[1]
            # predict with each model on latest row of dataframe
            predictions[pred_name] = models[i].predict(pd.DataFrame(latest))[0]
            
        # append dict of predictions to dataframe
        walking_df = pd.concat([walking_df, pd.DataFrame.from_records([predictions])])

        # Apply pipeline (inference_prep) on dataframe
        walking_df = pipeline_inference_prep(cfg=cfg).fit_transform(walking_df)

        # progress bar 
        pbar.update(100/hours)
        pbar.set_description('Predict weather for: ' + str(walking_df['timestamp'].iloc[-1]))

        # exit while loop once end date of incremental inference is reached
        if walking_df['timestamp'].iloc[-1] == end_date:
            break

    pbar.close()

    return walking_df
