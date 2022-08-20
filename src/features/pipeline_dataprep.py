import sys
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from config import data_config
from features.pipeline_dataprep_classes import Prepare
from features.pipeline_dataprep_classes import Acceleration
from features.pipeline_dataprep_classes import Velocity
from features.pipeline_dataprep_classes import InsertLags
from features.pipeline_dataprep_classes import Debugger
from features.pipeline_dataprep_classes import Times
from features.pipeline_dataprep_classes import Split
from features.pipeline_dataprep_classes import Scaler
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
        ("split", Split(n_splits = cfg.cv.n_splits)), 
        ("times", Times()),
        ('velocity', Velocity(vars=cfg.transform.vars, diff=cfg.diff.diff)),   
        ('acceleration', Acceleration(vars=cfg.transform.vars, diff=cfg.diff.diff)),  # diff of 1 row between 2 velos
        ('lags', InsertLags(diff=cfg.diff.lags)),
        #('debug', Debugger()),
        ('scale', Scaler(std_target=False)),  
        ('cleanup', Prepare(target = cfg.model.target, vars=cfg.model.predictors))
        ])

    return pipe

'''
variable explanation

velo_1: t_1 - t_0 der Variable
velo_2: t_2 - t_0 der Variable
acc_1: Differenz von 2 aufeinander folgenden velo_1
acc_2: Differenz von t_2 - t_0 von velo_2
(-> es fehlen Differenzen! z.B. Diff t_1 - t_0 von velo_2)
    Grund:
    - dummy.append(pd.DataFrame(data[k].iloc[:,col_indices].diff(periods = i).diff(periods = i)))
    - man iteriert durch i und i am Ende und nicht durch z.B i und j
    - Lösung: nested loop oder https://stackoverflow.com/questions/17006641/single-line-nested-for-loops

    - Folgeproblem: Wie benennt man die Variablen?
        - 'acc_2_1' -> Diff t_2 - t_0 von velo_1

    - Nur Beschleunigung zwischen 2 DIREKT aufeinanderfolgenden velos betrachten?
        - siehe Kommentar: "diff of 1 row between 2 velos"
        - Argument: Bei Diff von 2 weit entfernten Velos geht Information verloren
        - -> dummy.append(pd.DataFrame(data[k].iloc[:,col_indices].diff(periods = i).diff(periods = 1)))

Lags: um wieviele Reihen die vars verschoben wurden
'''

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


# All folds are here
train_folds = data['train']
test_folds = data['test']
train_std_folds = data['train_std']
test_std_folds = data['test_std']

# the last fold -> complete data. Last key is same for raw and std. data
last_train_key = list(data['train'])[-1]
last_test_key = list(data['test'])[-1] 

# full dataframes
train = data['train'][last_train_key]
test = data['test'][last_test_key]
train_std = data['train_std'][last_train_key]
test_std = data['test_std'][last_test_key]
pd_df = data['pd_df'] # train + test

check = data['train_std']['train_fold_2']
print(check)

print(type(train))
#print(train.dtypes)
print(train.head(15))
print(test)
print(pd_df)
#print(list(pd_df))

print(list(train_std))
print(train_std.iloc[0:15,0:9])

print(test_std.iloc[0:15,0:9])
####


# index = False -> no row names
train.to_csv(r'A:\Projects\ML-for-Weather\data\processed\train.csv', 
             header=True, index=False)
test.to_csv(r'A:\Projects\ML-for-Weather\data\processed\test.csv', 
             header=True, index=False)
train_std.to_csv(r'A:\Projects\ML-for-Weather\data\processed\train_std.csv', 
             header=True, index=False)
test_std.to_csv(r'A:\Projects\ML-for-Weather\data\processed\test_std.csv', 
             header=True, index=False)

