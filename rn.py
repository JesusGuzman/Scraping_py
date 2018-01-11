# -*- coding: utf-8 -*-
import numpy as np
 
####
# Vamos a crear una pequeña red neuronal que compute
# los valores de la tabla XOR
####
 
##
# Una red neuronal al final son:
#   * Una capa de inputs
#   * Las conexiones de TODOS LOS INPUTS con TODOS LOS OUTPUTS ( W en la imagen )
#   * Una capa de outputs
# La red neuronal se encarga de computar los INPUTS con las CONEXIONES
# para predecir los OUTPUTS
# Para que sea realmente funcional, necesitaremos saber
# cuanto afecta cada conexión al output
##
 
##Primera parte, generemos un input que sea una matriz
## Matriz de 2x4
## 0 0
## 0 1
## 1 0
## 1 1
INPUT_X = np.array([[0,0],[0,1],[1,0],[1,1]])
print "INPUT_X:\n", INPUT_X
 
##Luego la matriz de los resultados esperados
## en nuestro caso
## 0
## 1
## 1
## 0
EXPECTED_RESULT = np.array([[0,1,1,0]]).T # queremos en formato columna
print "EXPECTED_RESULT:\n", EXPECTED_RESULT
## Resultado: la tabla de verdad de la operación XOR
print "RESULT:\n", np.append(INPUT_X,EXPECTED_RESULT, axis=1)
 
## Para aprender, la red neuronal debe computar su predicción
## calcular el error, corregir los pesos y volver a computar
## Como tenemos DOS inputs y un OUTPUT, lo que buscamos es conectar
## cada INPUT con el OUTPUT, cada INPUT debe tener un peso, es decir
## cada INPUT modifica MAS o MENOS el output segun su peso.
 
# Esta es la función de activacion
# que nos permite modelar problemas NO lineales
# Esta mapea cualquier valor a entre 0 y 1
# Aprovechamos la misma función para devolver su derivada, luego lo explicamos.
def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))
 
#Seteamos una SEED que nos ayudarà a que los numeros esten
# distribuidos de forma random, pero siempre igual para entender
# en qué afectan los cambios que realizamos
np.random.seed(0)
 
# Inizializamos las CONEXIONES con una media a 0
# Esta es la matriz que irà "aprendiendo"
# La matriz resultando es de 2 x 1 ya que tenemos dos INPUTS y
# un OUTPUT
## **Ahora creamos una matriz de 2x3 para las conexiones de la CAPA0 a CAPA1
SYN0 = 2*np.random.random((2,3)) - 1
## **Y las conexiones de la CAPA1 a la CAPA2 que es el output
SYN1 = 2*np.random.random((3,1)) - 1
 
# Vamos a preparar iteraciones para aprender
# Probad a cambiar este numero para ver en qué afecta
for i in xrange(20000):
    # Primero, computamos con nuestra red, a este paso lo llamamos FORWARD
    # que resultado tenemos con los pesos
    # inicializados de forma random,
    l0 = INPUT_X
    #Multiplicamos los INPUTS con las CONEXIONES
    l1 = sigmoid(np.dot(l0, SYN0))
    ## **en l1 tenemos los resultados de la primera activación
    ## **debemos mover esos datos a la siguiente capa
    l2 = sigmoid(np.dot(l1, SYN1))
 
    #Computamos el error de la capa final
    l2_error = EXPECTED_RESULT - l2
 
    ## ** Computamos la diferencia con la derivada
    ## ** esto nos da el valor que debemos añadir
    ## ** a los pesos para "aprender"
    l2_delta = l2_error * sigmoid(l2, True)
 
    ## ** Ahora debemos saber qué parte del error acumulado
    ## ** es por culpa de las primeras conexiones
    l1_error = np.dot(l2_delta, SYN1.T)
 
    ## ** Finalmente, computamos qué valor debemos ajustar
    ## ** en las conexiones de SYN1
    l1_delta = l1_error * sigmoid(l1, True)
 
    ## Las primeras conexiones las corregimos
    ## con l1_delta * los inputs
    SYN0 += np.dot(l0.T, l1_delta)
    ## Las segundas conexiones las corregimos con
    ## l2_delta * los inputs de la capa intermedia
    SYN1 += np.dot(l1.T, l2_delta)
    if (i % 1000) == 0 :
        print "Error:" + str(np.mean(np.abs(l2_error)))
 
print l2
