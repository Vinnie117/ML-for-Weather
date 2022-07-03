# To do - short term

- warning with svm: Feature names unseen at fit time: -> Feature names seen at fit time, yet now missing:
    - check column names
    - -> csv data itself has no column names
- need DVC? -> keep track of what data was used in each experiment
- compare relative performance of using later lags
    - plot -> x: latest lag used, y: score on test data
- UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error.

- Implement a multiple models
    -> main motivation: establish a structure to track and log multiple models (e.g. performance, data used etc.)
    - crucial aspect to be able to track experiments
    - MLFlow. DVC -> research and compare! https://www.youtube.com/watch?v=W2DvpCYw22o 
- plot learning / loss curves?
    - https://scikit-learn.org/stable/modules/learning_curve.html
    - error curves by number of training data -> in order to see im performance increases!
- create (lagged) trend and seasonality variables
    - moving averages
    - 6h MA? 3h MA?
- create z-score variables?
    - https://twitter.com/mattrowsboats/status/1514293331278372876 



# To do - long term
- structure for multiple models / experiments
    - how are they (data / parameters) logged and tracked?
    - how should the folder structure be?
    - main.py with a main()-method as a main entry point for the app?
- Sub config file for each group of data to be used?
    - with hydra
        - https://www.youtube.com/watch?v=tEsPyYnzt8s
        - 24:30
    - or how to do it with another way?
        - data to be used must be tracked
- Deployment
    - https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166
    - https://towardsdatascience.com/deploying-machine-learning-models-into-a-website-using-flask-8582b7ce8802
    - https://www.natasshaselvaraj.com/ml-model-deployment/
    - https://www.kdnuggets.com/2020/05/build-deploy-machine-learning-web-app.html
- MLFlow / DVC
- Incorporate Explainable AI
    - LIME
    -SHaP
- Think about model evaluation
- Think about Exploratory Data Analysis


# Sources
- https://developer.nvidia.com/blog/global-ai-weather-forecaster-makes-predictions-in-seconds/ 
- https://github.com/jweyn/DLWP 
- https://courses.nvidia.com/courses/course-v1:DLI+S-ES-01+V1/

# chain multiple functions
- https://stackoverflow.com/questions/20454118/better-way-to-call-a-chain-of-functions-in-python
