# To do - short term

Wrap model in a simple API with FastAPI (inference, inference pipeline?)
- https://testdriven.io/blog/fastapi-machine-learning/
- https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-with-fastapi-docker-and-github-actions-13374cbd638a
- Build an inference pipeline / predict function


- Complete the docs!
    - Overall approach: describe main() and examine which function invokes another function
        - here, also give a description of functions
        also make a new Excalidraw
    - Write about the reason for the walking inference
    - write about procedure to do, when adding a new feature -> what pipelines / classes are affected
    - description of directories




- give confidence of one single inference point - how?
- compare relative performance of using later lags
    - plot -> x: latest lag used, y: score on test data
- plot learning / loss curves?
    - https://scikit-learn.org/stable/modules/learning_curve.html
    - error curves by number of training data -> in order to see im performance increases!
- create (lagged) trend and seasonality variables
    - moving averages
    - 6h MA? 3h MA?
- create z-score variables?
    - https://twitter.com/mattrowsboats/status/1514293331278372876 
- config management? What about multiple config files (in hydra)?



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
- Incorporate Explainable AI
    - LIME
    -SHaP
- Think about model evaluation
- Think about Exploratory Data Analysis


# Sources
- https://developer.nvidia.com/blog/global-ai-weather-forecaster-makes-predictions-in-seconds/ 
- https://github.com/jweyn/DLWP 
- https://courses.nvidia.com/courses/course-v1:DLI+S-ES-01+V1/

