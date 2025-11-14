from fastapi import FastAPI

app = FastAPI()


@app.get("/predict")
def read_item(q: str):
    return [0, 1, 2]
