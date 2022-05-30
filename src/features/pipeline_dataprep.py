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
        ('acceleration', Acceleration(vars=cfg.transform.vars, diff=cfg.diff.diff)),  # diff of 1 row between 2 velos
        ('lags', InsertLags_2(vars=cfg.transform.vars, diff=cfg.diff.lags)),  
        #('debug2', Debugger()),
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

train_2 = data_2['train']
test_2 = data_2['test']
pd_df_2 = data_2['pd_df']

print(train_2)
print(test_2)
print(pd_df_2)
#print(train_2.columns.tolist())


print(pd_df_2.columns.tolist())


####

np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\train_array.csv', train, delimiter=",", fmt='%s')
np.savetxt(r'A:\Projects\ML-for-Weather\data\processed\test_array.csv', test, delimiter=",", fmt='%s')