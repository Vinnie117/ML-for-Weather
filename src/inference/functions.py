import logging
import mlflow


def model_loader():
    '''
    This function automatically returns the best models (run from e.g. GridSearchCV) in a dict
    '''

    logging.info('FETCH MODELS FROM MLFLOW DIRECTORY')

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

        # load and assign all models as a PyFuncModel
        models[model_name] = mlflow.pyfunc.load_model('runs:/' + i + '/best_estimator')

    logging.info('THE MODEL IDs USED ARE: \n {models}'.format(models = models))
    return models