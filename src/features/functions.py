import pandas as pd

def clean(data):
    data = data.drop(columns=["station_id", "dataset"])
    data = data.pivot(index="date", columns="parameter", values="value").reset_index()
    data = data.rename(columns={'temperature_air_mean_200': 'temperature', 'cloud_cover_total': 'cloud_cover',
                        'wind_speed': 'wind_speed'})

    # reorder
    data.insert(1, 'temperature', data.pop('temperature'))

    return data