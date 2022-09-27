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

Plan Inference
1. 1 Model pro Target trainieren
2. Jedes Model (pro Target) sein Target für die NÄCHSTE Zeile in pd_df predicten lassen
    - pd_df muss alle Underlyings, statt 1 Target beinhalten
3. Nachdem die predicted Underlyings da sind, die Transforms davon in pd_df erstellen
    - mit einer eigenen Pipeline?
4. Schritte 2. und 3. so lange bis zum Zielzeitpunkt durchführen
5. Inference am Ende soll nur Zeit als Input entgegenehmen
    - -> bis zu diesem Timestamp wird trainiert und appended. Dann nochmal komplett trainiert.

Dieser Ansatz ist auch gut für ein potenzielles Retraining der Modelle





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
