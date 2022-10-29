import mlflow
import pandas as pd


# tags."mlflow.runName" LIKE "iteration11_run_number%"
df = mlflow.search_runs(['3'], filter_string="tags.mlflow.runName ILIKE '%XGB, target:%'")

test = df[["tags.mlflow.runName"]]
print(test.value_counts())


print(df[["tags.mlflow.runName"]])
print(df)
print(df[['run_id', 'tags.mlflow.runName', 'metrics.adjusted_r2']])


####
# sort by adjusted_r2,  then take  first element ( = minimum) in each runName group:
df4 = df.sort_values("metrics.adjusted_r2").groupby("tags.mlflow.runName", as_index=False).first()
print(df4)
print(df4[['run_id', 'tags.mlflow.runName', 'metrics.adjusted_r2']])

####
# now load all different models

# for i, j in zip(run_id, runName) -> dann str.split() und den Namen zusammen setzen -> assignment zusammensetzen -> in einem dict sammeln
model_names = []
run_ids = []
models = {}
for i, j in zip(df4['run_id'], df4['tags.mlflow.runName']):
    
    print('i is: ', i)
    print('j is: ', j)

    # construct model_name
    var = j.split('target: ')[1]
    model_name = "model_" + var
    model_names.append(model_name)

    # fetch run_id
    run_ids.append(i)

    # assign model
    models[model_name] = mlflow.pyfunc.load_model('runs:/' + i + '/best_estimator') 




print(model_names)
print(run_ids)
print(models)


model_temperature = mlflow.pyfunc.load_model('runs:/c8e2ca3172b64f1999116b4a8b290e7e/best_estimator')

print(model_temperature)


#############################################################################
# es ist 'metrics.adjusted_r2' drin -> danach filtern

# delelte all runs that are run_name = None
# For loop: For each unique run_name fetch the run with the lowest adj_r2
# For loop: For each run_name = XGB, target:{cfg...} fetch the run with the lowest adj_r2
# For loop: ... run_name contains... -> LIKE %abc%
# Print("the following run Ids are used for Inference: ...")


    # model_temperature = 'runs:/c8e2ca3172b64f1999116b4a8b290e7e/best_estimator'
    # model_cloud_cover = 'runs:/144c1be3dab346a19c95605c46c675f9/best_estimator'
    # model_wind_speed = 'runs:/fa8aed07ece5402f923ea24776d5b405/best_estimator'


print('END')