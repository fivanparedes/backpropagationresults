import numpy as np
import matplotlib.pyplot as plt
import pandas


# Función de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada de la función de activación
def sigmoid_derivada(x):
    return x * (1 - x)

# Leer el archivo CSV que contiene los datos de entrada y salida
df = pd.read_csv("datos.csv")

# Filtrar y limpiar el dataframe para eliminar filas con valores faltantes o inválidos
df = df.dropna()
df = df[(df["Horas de estudio"] >= 0) & (df["Porcentaje de asistencia"] >= 0) & (df["Promedio de calificaciones previas"] >= 0) & (df["Nota final"] >= 0)]

# Obtener las tres columnas numéricas como un array de numpy
X = df[["Horas de estudio", "Porcentaje de asistencia", "Promedio de calificaciones previas"]].to_numpy()
y = df[["Nota final"]].to_numpy()

# Dividir el dataframe en conjuntos de entrenamiento y verificación (80-20%) usando solo pandas y no sklearn
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

X = train_df[["Horas de estudio", "Porcentaje de asistencia", "Promedio de calificaciones previas"]].to_numpy()
Y = train_df[["Nota final"]].to_numpy()


# Semilla para los números aleatorios
np.random.seed(1)

# Inicialización de los pesos
n0 = input('Ingrese las neuronas de la capa oculta (defecto 3): ')
if n0 == '':
    n0 = 3
else:
    n0 = int(n0)

# 3xn0 matriz de pesos sinápticos de la capa de entrada a la capa oculta compuesta por n0 neuronas
w0 = 2 * np.random.random((3, n0)) - 1
# n0x1 matriz de pesos sinápticos de la capa oculta a la capa de salida compuesta por 1 neurona
w1 = 2 * np.random.random((n0, 1)) - 1

# Factor de aprendizaje
alpha = input('Ingrese el factor de aprendizaje (defecto 0.1): ')
if alpha == '':
    alpha = 0.1
else:
    alpha = float(alpha)

# Lista para guardar los errores
error = []

# Iteraciones
n = input('Ingrese el número de iteraciones (defecto 10000): ')
if n == '':
    n = 10000
else:
    n = int(n)

# Entrenamiento
for i in range(n):
    # Forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    # Backpropagation
    l2_error = y - l2
    l2_delta = l2_error * sigmoid_derivada(l2)

    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid_derivada(l1)

    # Actualización de los pesos
    w1 += l1.T.dot(l2_delta) * alpha
    w0 += l0.T.dot(l1_delta) * alpha

    # Guardar el error
    error.append(np.mean(np.abs(l2_error)))

# Gráfica del error
plt.plot(error)
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.show()

# Etapa de verificación
# Conjunto de verificación de entrada
X_test = test_df[["Horas de estudio", "Porcentaje de asistencia", "Promedio de calificaciones previas"]].to_numpy()

# Conjunto de verificación de salida
y_test = test_df[["Nota final"]].to_numpy()

# Forward propagation
l0 = X_test
l1 = sigmoid(np.dot(sigmoid(np.dot(l0, w0)), w1))

mean_acierto = 0
# Imprimir en una lista la salida obtenida, la salida esperada y el porcentaje de error
for i in range(len(l1)):
    mean_acierto += round(1 - np.abs(l1[i][0] - y_test[i][0]), 2) * 100
    print('Entrada ingresada: ', repr(X_test[i]), 'Salida esperada: ', y_test[i],
          'Salida obtenida: ', l1[i], '% de Precisión: ',
          round(1 - np.abs(l1[i][0] - y_test[i][0]), 2) * 100, '%')

print('Estructura de la red: ', X.shape[1], 'x', n0, 'x', y.shape[1],
      'Porcentaje de precisión promedio: ', mean_acierto / len(l1), '%')

# Conjunto de prueba
horas_estudio = input('Ingrese las horas de estudio: ')
porcentaje_asistencia = input('Ingrese el porcentaje de asistencia: ')
promedio_calificaciones = input('Ingrese el promedio de calificaciones: ')

# Pasar a float los datos de prueba
horas_estudio = float(horas_estudio)
porcentaje_asistencia = float(porcentaje_asistencia)
promedio_calificaciones = float(promedio_calificaciones)

X_test = np.array([[horas_estudio, porcentaje_asistencia, promedio_calificaciones]])

resultado = sigmoid(np.dot(sigmoid(np.dot(X_test, w0)), w1))

# Salida
print('Pesos sinápticos de la capa de entrada a la capa oculta:')
print(w0)

print('Pesos sinápticos de la capa oculta a la capa de salida:')
print(w1)

print('Numero de iteraciones: ', n)

print('Factor de aprendizaje: ', alpha)

for i in range(len(X_test)):
    print('Entrada: ', X_test[i], ' Salida: ',
          'Aprueba ({})'.format(resultado[i]) if resultado[i] > 0.55 else 'Reprueba ({})'.format(resultado[i]))

# TODO: Restringir el ingreso de datos por consola, horas_estudio debe ser positivo,
#  porcentaje_asistencia debe estar entre 0 y 1, promedio_calificaciones debe estar entre 0 y 1

