import mlflow
import pandas as pd


# tags."mlflow.runName" LIKE "iteration11_run_number%"
df = mlflow.search_runs(['3'], filter_string="tags.mlflow.runName != 'None'")

test = df[["tags.mlflow.runName"]]
print(test.value_counts())


print(df[["tags.mlflow.runName"]])
print(df)
print(df[['run_id', 'metrics.adjusted_r2']])

# es ist 'metrics.adjusted_r2' drin -> danach filtern
print(list(df))

# delelte all runs that are run_name = None
# For loop: For each unique run_name fetch the run with the lowest adj_r2
# For loop: For each run_name = XGB, target:{cfg...} fetch the run with the lowest adj_r2
# For loop: ... run_name contains... -> LIKE %abc%
# Print("the following run Ids are used for Inference: ...")


    # model_temperature = 'runs:/c8e2ca3172b64f1999116b4a8b290e7e/best_estimator'
    # model_cloud_cover = 'runs:/144c1be3dab346a19c95605c46c675f9/best_estimator'
    # model_wind_speed = 'runs:/fa8aed07ece5402f923ea24776d5b405/best_estimator'


print('END')