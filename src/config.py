from dataclasses import dataclass
import string

@dataclass
class vars_old:
    temp: str
    cloud: str

@dataclass
class vars_new:
    temp: str
    cloud: str

@dataclass
class data:
    path: str

@dataclass
class diff:
    lags: int
    diff: int

@dataclass
class transform:
    vars: str

@dataclass
class model:
    target: str
    predictors: str

@dataclass
class cv:
    n_splits: int

@dataclass
class elastic_net:
    alpha: float
    l1_ratio: float

@dataclass
class svm:
    C: int
    gamma: float
    kernel: string

@dataclass
class data_config:
    vars_old: vars_old
    vars_new: vars_new
    data: data
    transform: transform
    diff: diff
    model: model
    cv: cv
    elastic_net: elastic_net
    svm: svm


