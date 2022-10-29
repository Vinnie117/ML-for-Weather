from fastapi import FastAPI
from src.features.pipeline_dataprep import data_loader, pipeline_feature_engineering, save
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from src.config import data_config
from src.models.XGB.training_xgboost import model_data_loader, train_xgb
import mlflow
from src.inference.inference import pipeline_features_inference, walking_inference, model_loader


# # start app in venv: uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# app = FastAPI()


# @app.get("/ping")
# def pong():
#     return {"ping": "pong!"}


def main_training(target):
    ''' Function for model training'''

    # load and prepare training data
    df = data_loader('training', cfg=cfg)
    dict_data = pipeline_feature_engineering(cfg = cfg, target = target).fit_transform(df) 

    # the last fold is complete data
    last_train_key = list(dict_data['train'])[-1]
    last_test_key = list(dict_data['test'])[-1] 

    # full dataframes
    train = dict_data['train'][last_train_key]
    test = dict_data['test'][last_test_key]
    train_std = dict_data['train_std'][last_train_key]
    test_std = dict_data['test_std'][last_test_key]

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


def main(cfg: data_config, training, inference):
    if training:
        for i in cfg.transform.vars:
            main_training(target = i)
    if inference:
        df = main_inference(cfg=cfg)
        return df


if __name__ == "__main__":
    
    # use configs
    initialize(config_path="src/conf", job_name="config")
    cfg = compose(config_name="config")
    cs = ConfigStore.instance()
    cs.store(name = 'data_config', node = data_config)

    # start program
    df = main(cfg = cfg, training = True, inference = False )

    #print(df[['year', 'month', 'day', 'hour', 'temperature']].tail(10))

    print('END')
