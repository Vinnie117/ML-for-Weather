import os
import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from sklearn.pipeline import Pipeline
from config import data_config
from training.classes_training_features import Prepare
from training.classes_training_features import Acceleration
from training.classes_training_features import Velocity
from training.classes_training_features import InsertLags
from training.classes_training_features import Debugger
from training.classes_training_features import Times
from training.classes_training_features import Split
from training.classes_training_features import Scaler
import logging


# Feature engineering
def pipeline_training_features(cfg: data_config, target):
    '''
    Pipeline for feature engineering of training data
    '''

    logging.info('PREPARE DATA FOR TRAINING')

    pipe = Pipeline([
        ("split", Split(n_splits = cfg.cv.n_splits)), 
        ("times", Times()),
        ('velocity', Velocity(vars=cfg.transform.vars, diff=cfg.diff.diff)),   
        ('acceleration', Acceleration(vars=cfg.transform.vars, diff=cfg.diff.diff)),  # diff of 1 row between 2 velos
        ('lags', InsertLags(diff=cfg.diff.lags)),
        #('debug', Debugger()),
        ('scale', Scaler(target = target, std_target=False)),  
        ('cleanup', Prepare(target = target, predictors=cfg.model.predictors, vars = cfg.transform.vars))
        ])
        
    return pipe