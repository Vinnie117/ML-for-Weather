from fastapi import FastAPI
from preprocessing.training_preproc import pipeline_training_preproc
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from src.config import data_config
from src.training.XGB.training_xgboost import train_xgb
from src.training.functions import model_data_loader
from src.inference.inference import pipeline_features_inference, walking_inference
from utils.functions import download, data_loader
from src.preprocessing.functions import save, dict_to_df
import logging

# # start app in venv: uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# app = FastAPI()


# @app.get("/ping")
# def pong():
#     return {"ping": "pong!"}


def main_training(target):
    ''' Function for model training of a single target variable'''

    # load and prepare training data
    df = data_loader('training', cfg=cfg)
    dict_data = pipeline_training_preproc(cfg = cfg, target = target).fit_transform(df) 

    # create dataframes
    train, test, train_std, test_std = dict_to_df(dict_data = dict_data)

    # save to database
    save(var=target, train=train, test=test, train_std=train_std, test_std=test_std)

    # model training with mlflow
    X_train, y_train, X_test, y_test = model_data_loader(target = target)
    train_xgb(cfg = cfg, target = target, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def main_inference(cfg: data_config):
    ''' Function for model inference '''

    # load data for inference
    df_inference = data_loader('inference',cfg=cfg)

    # make inference
    df = pipeline_features_inference(cfg=cfg).fit_transform(df_inference)
    df = walking_inference(cfg=cfg, walking_df=df, end_date=cfg.inference.end_date)

    return df


def main(cfg: data_config, api_download, training, inference):
    if api_download:
        download(cfg = cfg)
    if training:
        for i in cfg.transform.vars:
            main_training(target = i)
    if inference:
        df = main_inference(cfg=cfg)
        return df


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    # use configs
    logging.info('INITIALIZE CONFIGS')
    initialize(config_path="src/conf", job_name="config")
    cfg = compose(config_name="config")
    cs = ConfigStore.instance()
    cs.store(name = 'data_config', node = data_config)

    # start program
    df = main(cfg = cfg, api_download = True , training = False, inference = False )

    #print(df[['year', 'month', 'day', 'hour', 'temperature']].tail(10))

    logging.info('END OF APP')
