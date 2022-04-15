from dataclasses import dataclass

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
class data_config:
    vars_old: vars_old
    vars_new: vars_new
    data: data