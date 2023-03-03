import joblib

model = joblib.load("models/iris_svm_model.pkl")

def predict_iris(data:list) -> list:
    """
    Esta es la funcion que nos hara la prediccion de la clase
    """

    predictions = model.predict(data)
    return predictions.tolist()


