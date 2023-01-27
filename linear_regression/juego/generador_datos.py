import random
import pandas as pd

# Generar datos de ejemplo de juego
num_juegos = 100
numero_a_adivinar = [random.randint(1, 100) for i in range(num_juegos)]
intentos = [random.randint(1, 10) for i in range(num_juegos)]

datos = {'intentos': intentos, 'numero_correcto': numero_a_adivinar}
df = pd.DataFrame(datos)
df.to_csv('datos/juego.csv', index=False)


