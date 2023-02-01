import random
import pandas as pd
from modelo import entrenar_modelo
import pickle

# generar número aleatorio a adivinar kjldsaljksdakjlsdakjlsdakjlsdakjlsdajklsda
numero_a_adivinar = random.randint(1, 100)
regresion_lineal = pickle.load(open('modelos/modelo_regresion.pkl','rb'))

# hacer una predicción del número de intentos necesarios para adivinar el número correcto
prediccion_numero_de_intentos = int(regresion_lineal.predict([[numero_a_adivinar]]))
# iniciar juego
intentos = 0
while True:
    intentos += 1
    numero_adivinado = int(input("Adivina el número (1-100) deberia tomarte {} intentos: ".format(prediccion_numero_de_intentos)))

    if numero_adivinado == numero_a_adivinar:
        print("¡Felicidades! Adivinaste el número en", intentos, "intentos.")
        break
    elif numero_adivinado > numero_a_adivinar:
        print("El número es más pequeño.")
    else:
        print("El número es más grande")

# guardar datos de juego en un archivo csv
datos = {'intentos': [intentos], 'numero_correcto': [numero_a_adivinar]}
df = pd.DataFrame(datos)
df.to_csv('datos/simulacion.csv', mode='a', header=False,index=False)

entrenar_modelo('datos/simulacion.csv')


