from dataclasses import dataclass

@dataclass
class vars_old:
    temp: str
    cloud: str

@dataclass
class vars_new:
    temp: str
    cloud: str
    wind: str

@dataclass
class data:
    path: str

@dataclass
class predictors:
    vars: str

@dataclass
class diff:
    lags: int
    velo: int
    acc: int

@dataclass
class data_config:
    vars_old: vars_old
    vars_new: vars_new
    data: data
    predictors: predictors
    diff: diff