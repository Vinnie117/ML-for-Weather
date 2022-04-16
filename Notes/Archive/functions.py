import pandas as pd

def clean(data, old, new):
    data = data.drop(columns=["station_id", "dataset"])
    data = data.pivot(index="date", columns="parameter", values="value").reset_index()

    # better column names
    for i in range(len(old)):
        data = data.rename(columns={old[i]: new[i]})

    # reorder
    data.insert(1, 'temperature', data.pop('temperature'))

    return data