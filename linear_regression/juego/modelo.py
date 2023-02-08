from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

def entrenar_modelo(archivo_csv):
    # importar datos del archivo csv
    datos = pd.read_csv(archivo_csv)


    X = datos['numero_correcto'].values.reshape(-1,1)
    y = datos['intentos'].values.reshape(-1,1).ravel()
    


    # dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # entrenar el modelo de regresión lineal
    regresion_lineal = LinearRegression()
    logistic_regression=LogisticRegression(max_iter=1000)

    logistic_regression.fit(X_train, y_train)
    regresion_lineal.fit(X_train, y_train)

    # guardar el modelo entrenado en un archivo
    pickle.dump(regresion_lineal, open('modelos/modelo_regresion.pkl', 'wb'))
    pickle.dump(logistic_regression, open('modelos/logistic_regresion.pkl', 'wb'))


