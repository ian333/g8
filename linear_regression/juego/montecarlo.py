import random
import pandas as pd
import numpy as np
import pickle
from modelo import entrenar_modelo
#entrenar_modelo('datos/simulacion.csv')


# cargar modelo entrenado
#regresion_lineal = pickle.load(open('modelos/modelo_regresion.pkl','rb'))
#logistic_regression=pickle.load(open('modelos/logistic_regresion.pkl','rb'))
def juego_simulado():
    for _ in range(300):

        # generar número aleatorio a adivinar
        numero_a_adivinar = random.randint(1, 100)
        # hacer una predicción del número de intentos necesarios para adivinar el número correcto
        prediccion_numero_de_intentos = 3
        # iniciar juego
        intentos = 0
        aux_menor=1
        aux_mayor=101
        while True:

            intentos += 1
            print(f"Rangos({aux_menor},{aux_mayor})")
            numero_adivinado = random.randint(aux_menor, aux_mayor)
            #print(f"Numero ingresado {numero_adivinado}")
            #print(f"Numero a adivinar {numero_a_adivinar}")
            #print("Adivina el número (1-100) deberia tomarte {} intentos: ".format(prediccion_numero_de_intentos))

            if numero_adivinado == numero_a_adivinar:
                print("¡Felicidades! Adivinaste el número",numero_a_adivinar," en", intentos, "intentos y la prediccion era",prediccion_numero_de_intentos)
                break
            elif numero_adivinado > numero_a_adivinar:
                print("El número es más pequeño que ",numero_adivinado)
                aux_mayor=numero_adivinado-1
                print("menor",aux_menor,numero_a_adivinar,numero_adivinado)
            else:
                aux_menor=numero_adivinado+1
                print("El número es más grande que ",numero_adivinado)
                print("mayor",aux_mayor,numero_a_adivinar,numero_adivinado)
        datos.append([intentos,numero_a_adivinar])
    return datos

# guardar datos de juego en un archivo csv
datos = []

datos = juego_simulado()
df = pd.DataFrame(datos, columns=["intentos","numero_correcto"])
df.to_csv('linear_regression/datos/simulacion.csv', mode='w', header=True,index=False)

mean = df["intentos"].mean()
std = df["intentos"].std()
print("Media de intentos:",mean)
print("Desviación estandar:",std)

