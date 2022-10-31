import mlflow
import pandas as pd


def model_loader():
    '''
    This function automatically load the best model (run from e.g. GridSearchCV)
    '''

    # search mlflow experiments by tag runName
    df = mlflow.search_runs(['3'], filter_string="tags.mlflow.runName ILIKE '%XGB, target:%'")

    # sort by adjusted_r2,  then take  first element ( = minimum) in each runName group:
    df = df.sort_values("metrics.adjusted_r2").groupby("tags.mlflow.runName", as_index=False).first()

    # now load all different models into a dict
    models = {}
    for i, j in zip(df['run_id'], df['tags.mlflow.runName']):
        
        # construct model_name
        var = j.split('target: ')[1]
        model_name = "model_" + var 

        # assign model
        models[model_name] = mlflow.pyfunc.load_model('runs:/' + i + '/best_estimator')
        print(model_name + ' has ID: ' + i)

    return models


models = model_loader()
print(models)







print('END')