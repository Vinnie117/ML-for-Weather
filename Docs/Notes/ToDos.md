# To do - short term

Wrap model in a simple API with FastAPI (inference, inference pipeline?)
- https://testdriven.io/blog/fastapi-machine-learning/
- https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-with-fastapi-docker-and-github-actions-13374cbd638a
- Build an inference pipeline / predict function

- how should inference work? giving time only? Or time and variables (if they exist) from download
    - alle Features bis dahin predicten? 
        - zu viel Rechenaufwand?
        - nur die underlying variables predicten und den Rest transformieren / laggen?
    - alle target values bis dahin berechnen?
        - das Modell ist nur auf Target values trainiert
    - nur Point Prediction möglich?

Plan
1. Mit XGBoost den subsequently den nächsten Tag predicten. Das für alle Underlyings machen
    - der vom Model aus möglich ist?
    - Anhand der möglichen Daten aus der DWD API?
    - -> beides kann / sollte (?) aufeinanderfallen, sodass immer vom Model aus der nächste Punkt predicted wird.
    - die notwendigen Trainingsdaten erstellen -> mit jedem Underlying als Target
2. die benötigten XGBs für jedes underlying target trainieren
3. das trainierte xgb auf die letzte Reihe von pd_df anwenden -> Inference für jeweils den nächsten Datenpunkt predicten. 
    - Mit while Schleife hochzählen bis zum Enddatum
    - Predictions sammeln
4. ein gejointes DF erstellen: tatsächliche + predictete Werte (nur Underlying)
5. Alle Transformations / Lags der predicted Underlyings mithilfe der Pipeline erstellen
    - selbe Struktur wie nach dem Download von der DWD API simulieren?
6. Ein finales XGBoost nutzen, um anhand dieses gejointen DFs das Target zum Zieldatum zu predicten
    - Ziel: Man soll am Ende nur ein Datum angeben müssen!





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
- Deployment (e.g. with FastAPI or RestAPI)
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
