import os
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
import logging

def data_loader(data, cfg: data_config):

    use = list(cfg['data'].keys())[list(cfg['data'].values()).index(cfg['data'][data])]   
    logging.info('LOAD DATA FOR {use} FROM DIRECTORY'.format(use = use.upper()))

    # load data (make this better! function arg should directly reference cfg)
    try:
        df_raw = pd.read_csv(cfg['data'][data])
    except:
        print("Function argument type must be of available type in config -> data")

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
def pipeline_feature_engineering(cfg: data_config, target):
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

'''
variable explanation

velo_1: t_1 - t_0 der Variable
velo_2: t_2 - t_0 der Variable
acc_1: Differenz von 2 aufeinander folgenden velo_1
acc_2: Differenz von t_1 - t_0 von velo_2 (auch 2 aufeinander folgenden velos, aber hier: velo_2)
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



def save(var, train, test, train_std, test_std):
    ''' Save (standardized) training and test data to folder ./data_dvc/processed
    @param target: The target variable of interest
    @param train: the training data to be saved
    @param test: the test data to be saved
    
    '''

    logging.info('SAVING DATA TO DIRECTORY')

    dir_name = os.path.join(os.getcwd(), 'data_dvc', 'processed') 
    base_filename_train = 'train_' + var
    base_filename_test = 'test_' + var
    base_filename_train_std = 'train_std_' + var
    base_filename_test_std = 'test_std_' + var
    format = 'csv'

    file_train = os.path.join(dir_name, base_filename_train + '.' + format)
    file_test = os.path.join(dir_name, base_filename_test + '.' + format)
    file_train_std = os.path.join(dir_name, base_filename_train_std + '.' + format)
    file_test_std = os.path.join(dir_name, base_filename_test_std + '.' + format)

    train.to_csv(file_train, header=True, index=False)
    test.to_csv(file_test, header=True, index=False)
    train_std.to_csv(file_train_std, header=True, index=False)
    test_std.to_csv(file_test_std, header=True, index=False)


def dict_to_df(dict_data):
    '''
    This function creates the complete dataframes after preprocessing. It specifically
    works with a dict of dicts containing data folds
    '''
    logging.info('FETCH DATAFRAMES FROM DICTIONARY')

    # the last fold is complete data
    last_train_key = list(dict_data['train'])[-1]
    last_test_key = list(dict_data['test'])[-1] 

    # full dataframes
    train = dict_data['train'][last_train_key]
    test = dict_data['test'][last_test_key]
    train_std = dict_data['train_std'][last_train_key]
    test_std = dict_data['test_std'][last_test_key]

    return train, test, train_std, test_std



if __name__ == "__main__":

    # Use Compose API of hydra 
    initialize(config_path="..\conf", job_name="config")
    cfg = compose(config_name="config")
    #print(OmegaConf.to_yaml(cfg))

    # Use instance of config dataclass
    cs = ConfigStore.instance()
    cs.store(name = 'data_config', node = data_config)

    df = data_loader('training', cfg=cfg)

    pipeline = pipeline_feature_engineering(cfg = cfg)
    dict_data = pipeline.fit_transform(df) 


    # All folds are here
    train_folds = dict_data['train']
    test_folds = dict_data['test']
    train_std_folds = dict_data['train_std']
    test_std_folds = dict_data['test_std']

    # the last fold -> complete data. Last key is same for raw and std. data
    last_train_key = list(dict_data['train'])[-1]
    last_test_key = list(dict_data['test'])[-1] 

    # full dataframes
    train = dict_data['train'][last_train_key]
    test = dict_data['test'][last_test_key]
    train_std = dict_data['train_std'][last_train_key]
    test_std = dict_data['test_std'][last_test_key]
    pd_df = dict_data['pd_df'] # train + test

    check = dict_data['train']['train_fold_2']


    save(target = cfg.model.target)

    # # Some prints for inspection
    # print(check)
    # print(list(check))

    # print(type(train))
    # print(train.dtypes)
    # print(train.head(15))
    # print(test)
    # print(pd_df)
    # print(list(pd_df))

    # print(list(train_std))
    # print(train_std.iloc[0:15,0:9])
    # print(test_std.iloc[0:15,0:9])


    print("END")

