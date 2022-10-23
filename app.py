from fastapi import FastAPI

# start app in venv: uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
app = FastAPI()


@app.get("/ping")
def pong():
    return {"ping": "pong!"}


def main():

    # put the whole ML workflow WITHOUT downloading the data in here
    # -> save() only needs train(_std) and test(_std) (in pipeline_dataprep())
    # -> rename data to dict_data in pipeline_dataprep.py
    # Think about folder structure first
    # Improve documentation!

    pass


if __name__ == "__main__":
    main()
