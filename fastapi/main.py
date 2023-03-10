from fastapi import FastAPI, File,UploadFile,Form
from typing import List,Optional
from schemas.predict_schemas import IrisData
from predict import predict_iris,predic_digits
from PIL import Image
import numpy as np
from utilities import transcribe

app=FastAPI()

@app.get("/")
def read_root():
    """
    Este endpoint devuelve un saludo de bienvenida.
    """
    return {"Hello": "¡Bienvenido a mi aplicación FastAPI!"}

@app.post("/svm/iris")
async def predict(data: List[IrisData]):
    data_list = [[d.sepal_length, d.sepal_width, d.petal_length, d.petal_width] for d in data]
    predictions = predict_iris(data_list)
    return {"predictions": predictions}

@app.get("/sumatoria/{number}")
def sumatoria(number:int):
    """
    A traves de una query se ingresa un numero el cual vamos a sumar con el numero 10

    """
    return {"Result":10+int(number)}

@app.post("/tree/image")
async def predict_digits(image:UploadFile=File(...)):

    with Image.open(image.file) as img:
        img = img.convert("G").resize((8, 8))
        data = np.asarray(img, dtype=np.float32).reshape(1, -1) / 16.0

    print(img)
    print(data)
    prediction = predic_digits(data)

    return {"Prediccion":prediction["prediction"]}



@app.post("/whisper/translate")
async def translate(audio:UploadFile=File(...)):

    pass


@app.post("/login/")
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}