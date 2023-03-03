import joblib
from typing import List
import numpy as np
model = joblib.load("models/iris_svm_model.pkl")
model_digits= joblib.load("models/digits_ExtratreeC_model.pkl")

def predict_iris(data:list) -> int:
    """
    Esta es la funcion que nos hara la prediccion de la clase
    """

    predictions = model.predict(data)
    return predictions

def predic_digits(data:np.ndarray):
    """
    Esta es la funcion que nos hara la prediccion de un numero segun una imagen
    """

    prediction=int(model_digits.predict(data)[0])

    return {"prediction": prediction}





