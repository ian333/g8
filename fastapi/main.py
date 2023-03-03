from fastapi import FastAPI, File,UploadFile,Form
from typing import List
from schemas.predict_schemas import IrisData
from predict import predict_iris
app=FastAPI()

@app.get("/")
def read_root():
    """
    Este endpoint devuelve un saludo de bienvenida.
    """
    return {"Hello": "¡Bienvenido a mi aplicación FastAPI!"}


@app.post("/svm/iris")
def predict(data: List[IrisData]):
    data_list = [[d.sepal_length, d.sepal_width, d.petal_length, d.petal_width] for d in data]
    predictions = predict_iris(data_list)
    return {"predictions": predictions}

@app.get("/sumatoria/{number}")
def sumatoria(number:int):
    """
    A traves de una query se ingresa un numero el cual vamos a sumar con el numero 10

    """
    return {"Result":10+int(number)}



@app.post("/svm/image")
def upload_file(image:UploadFile):

    return {"filename":image.filename}



@app.post("/login/")
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}