import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from config import data_config
from features.pipeline_dataprep_classes import Prepare
from features.pipeline_dataprep_classes import Acceleration
from features.pipeline_dataprep_classes import Velocity
from features.pipeline_dataprep_classes import InsertLags, InsertLags_2
from features.pipeline_dataprep_classes import Debugger
from features.pipeline_dataprep_classes import Times
from features.pipeline_dataprep_classes import Split
from hydra.core.config_store import ConfigStore
from hydra import compose, initialize


def data_loader(cfg: data_config):
    
    # load data
    df_raw = pd.read_csv(cfg.data.path)
    
    # clean up and prepare
    data = df_raw.drop(columns=['station_id', 'dataset'])
    data = data.pivot(index='date', columns='parameter', values='value').reset_index()
    
    # renaming
    for i in cfg.vars_old:
        data = data.rename(columns={cfg.vars_old[i]: cfg.vars_new[i]})
    
    # ordering
    data.insert(1, cfg.vars_new.temp, data.pop(cfg.vars_new.temp))

    return data


# Feature engineering
def feature_engineering(cfg: data_config):

    pipe = Pipeline([
        ("split", Split(test_size= cfg.model.split, shuffle = cfg.model.shuffle)), # -> sklearn.model_selection.TimeSeriesSplit
        ("times", Times()),
        ("lags", InsertLags(vars=cfg.transform.vars, diff=cfg.diff.lags)),
        ('velocity', Velocity(vars=cfg.transform.vars, diff=cfg.diff.velo)),   
        ('lagged_velocity', InsertLags(vars=cfg.transform.lags_velo, diff=cfg.diff.lagged_velo)),     # lagged difference = differenced lag
        ('acceleration', Acceleration(vars=cfg.transform.vars, diff=cfg.diff.acc)),                   # diff of rows (s) between 2 subsequent velos
        ('lagged_acceleration', InsertLags(vars=cfg.transform.lags_acc, diff=cfg.diff.lagged_acc)),   
        ('cleanup', Prepare(target = cfg.model.target, vars=cfg.model.predictors))
        ])

    return pipe


def feature_engineering_2(cfg: data_config):

    pipe = Pipeline([
        ("split", Split(test_size= cfg.model.split, shuffle = cfg.model.shuffle)), # -> sklearn.model_selection.TimeSeriesSplit
        ("times", Times()),
        ('velocity', Velocity(vars=cfg.transform.vars, diff=cfg.diff.diff)),   
        ('acceleration', Acceleration(vars=cfg.transform.vars, diff=cfg.diff.diff)),                   # diff of 1 day between 2 velos
        ('lags', InsertLags_2(vars=cfg.transform.vars, diff=cfg.diff.lags)),  
        ('debug2', Debugger()),
      #  ('debug2', Debugger()),
      #  ('cleanup', Prepare(target = cfg.model.target, vars=cfg.model.predictors))
        ])

    return pipe

# Use Compose API of hydra 
initialize(config_path="..\conf", job_name="config")
cfg = compose(config_name="config")
#print(OmegaConf.to_yaml(cfg))

# Use instance of config dataclass
cs = ConfigStore.instance()
cs.store(name = 'data_config', node = data_config)

df = data_loader(cfg=cfg)
pipeline = feature_engineering(cfg = cfg)
data = pipeline.fit_transform(df) 

train = data['train']
test = data['test']
pd_df = data['pd_df']

# print(train)
# print(test)
# print(pd_df)


####
# -> Nur einmal InsertLags(), vorher alle Variablen erstellen!!!
pipeline_2 = feature_engineering_2(cfg = cfg)
data_2 = pipeline_2.fit_transform(df) 


print(data_2['train'].columns.tolist())


print(pd_df.columns.tolist())


####

np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', train, delimiter=",", fmt='%s')
np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', test, delimiter=",", fmt='%s')