from fastapi import FastAPI

# start app in venv: uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
app = FastAPI()


@app.get("/ping")
def pong():
    return {"ping": "pong!"}
