#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jdariasl/ML_2020/blob/master/Labs/lab1/lab1_parte2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# **Recuerda que una vez abierto, Da clic en "Copiar en Drive", de lo contrario no podras alamancenar tu progreso**
# 
# Nota: no olvide ir ejecutando las celdas de código de arriba hacia abajo para que no tenga errores de importación de librerías o por falta de definición de variables.
# 

# In[ ]:


#configuración del laboratorio
# Ejecuta esta celda!
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().system('wget https://raw.githubusercontent.com/jdariasl/ML_2020/master/Labs/commons/utils/general.py -O general.py')
from general import configure_lab1_p2
configure_lab1_p2()
from lab1 import *
GRADER, x, y = part_2()


# ## Laboratorio 1 - Parte 2
# 
# **Regresión logística**
# 
# 

# ## Ejercicio 1: Contextualización del problema
# 
# En esta sesión de laboratorio, vamos a resolver un problema de clasificación. Los variables que vamos a usar ya se encuentran cargadas:
# 
# 

# In[ ]:


# tienes ya cargadas las siguientes variables:
print("conjunto de datos, muestra \n",x[range(10), :] )
print("")
print(" muestra de etiquetas a predecir \n", y[range(10)])


# In[38]:


#Ejercicio de Codigo
def clases_muestras_carac(X, Y):
    """Esta funcion es encargada retornar el numero clases, muestras 
        y caracteristicas del conjunto de datos X y Y

        X: matriz numpy con el conjunto de datos para entrenamiento
        Y: matriz numpy con el conjunto de etiquetas
        retorna:
            numero de clases (int/float)
            numero de muestras (int/float)
            numero de caracteristicas (int/float)
    """
    ##Pista: es de utilidad el metodo np.unique ?
    N,nf =
    clases = 
    
    return (N,nf,clases)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio1", clases_muestras_carac)


# En los problemas de clasificación, que lo permiten, es de utilidad visualizar los datos. De esta manera uno puede determinar que modelos o algortimos pueden tener mejor rendimiento. En la siguiente función, debera, graficar los datos usando la función [scatter](https://matplotlib.org/gallery/shapes_and_collections/scatter.html) de matplotlib.

# In[ ]:


#Ejercicio de Codigo
def scatter_plot(X, Y):
    """Esta funcion es encargada de graficar usando un scatter plot
       un problema de clasificacion.

        X: matriz numpy con el conjunto de datos para entrenamiento.
           esta debera ser usada para los ejes del grafico. puede asumir
           que solo va tener dos columnas
        Y: matriz numpy con el conjunto de etiquetas. Debera se usada
           para mostrar en diferentes colores, las etiquetas de cada una
           de las muestras
        retorna:
            grafica matplotlib
    """
    ## puedes accerder con plt al funcion adecuacada
    ## Pista: recuerda como indexar matrices
    ## Pista: recuerda el uso de np.ravel

    # para mostrar el grafico
    figure = plt.gcf()
    plt.show()
   
    return (figure)


# In[ ]:


## la funcion que prueba tu implementacion
# ignora los graficos que se muestran 
GRADER.run_test("ejercicio2", scatter_plot)


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿El problema es linealmente separable? justifique su respuesta 
respuesta_1 = "" #@param {type:"string"}


# ## Ejercicio 2: entrenamiento
# 
# En este laboratorio se va a realizar un procedimiento análogo al laboratorio anterior, pero con el modelo de *regresión logística* que sirve para resolver problemas de clasificación (en principio biclase).
# 
# Analice los siguientes métodos a la luz de la teoría vista para el modelo de regresión logística: 
# 
# 1. función de activación (<font color='blue'>sigmoidal</font>),
# 2.  modelo de regresión logística (<font color='blue'>logistic_regression</font>), 
# 3. potencia del polinomio 
# 4.  el cálculo del error en clasificación (<font color='blue'>error_logistic</font>)
# 5. el gradiente descendente. 
# 
# Luego de recordar estos conceptos. Complete la función sigmoidal:

# In[ ]:


#Ejercicio de Código
def sigmoidal(z):
    """Función de activación Sigmoidal

    z: es la varible a la que se le va aplicar el sigmoide.
       es un array numpy de uan sola dimension
    retorna: el valor del sigmiode

    """
    #Complete la siguiente línea con el código para calcular la salida de la función sigmoidal
    s = 
    
    return s


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio3", sigmoidal)


# El la siguiente celda se sugiere la implementación, de los siguientes conceptos:
# 
# 1. modelo de regresión logística (<font color='blue'>logistic_regression</font>), 
# 2. potencia del polinomio 
# 3.  el cálculo del error en clasificación (<font color='blue'>error_logistic</font>)
# 
# comprenda que hacen estas funciones y ejecute la celda para cargar las funciones, para porder usarlas

# In[ ]:


def logistic_regression(X, W):
    """calcula la regresión logistica
    X: los valores que corresponden a las caractersiticas
    W: son los pesos usadados para realizar la regresión
    retorna: valor estimado por la regresion
    """
    #Con np.dot se realiza el producto matricial. Aquí X (extendida) tiene dim [Nxd] y W es dim [dx1]
    Yest = np.dot(X,W)
    Y_lest = sigmoidal(Yest)
    #Se asignan los valores a 1 o 0 según el modelo de regresión logística definido
    pos = 0
    for tag in Y_lest:
        
        if tag > 0.5:
            Y_lest[pos] = 1
        elif tag < 0.5:
            Y_lest[pos] = 0
        
        pos += 1
    
    return Y_lest    #Y estimado: Esta variable contiene ya tiene la salida de sigm(f(X,W))

def potenciaPolinomio(X,grado):
    """calcula la potencia del polinomio
    X: los valores que corresponden a las caractersiticas
    grado: esl grado para realizar la potencia al polinomio
    retorna: el valor de X despues elevarlo al grado del polinimoo indicado
    """
    X2 = X.copy()
    
    if grado != 1:
        for i in range(2,grado+1):
            Xadd = X**i
            X2 = np.concatenate((X2, Xadd), axis=1)
    
    return X2

def error_logistic(Y_lest, Y):
    """calculo del error logistico
       Si es diferente el Y_estimado con el Y_real cuenta como un error
       Y_lest: numpy array con los valores de etiquetas estimadas
       Y:  numpy array  valor con los valores reales de las etiquetas
       retorna: error de clasificación -- numpy array
    """
    error = 0
    for ye, y in zip(Y_lest, Y):
        if ye != y:
            error += 1
    
    error = error/np.size(Y)

    return error


# De igual manera, debemos extender nuestro conjunto de datos. Comprende que hace  la siguiente celda de código y ejecutala. **Muy importante ejecutar SOLA UNA vez.**

# In[ ]:


#Aca llamamos la funcion creada anteriormente
# para obtener el numero muestras y caractersiticas
muestras,caracterisitcas,num_clases = clases_muestras_carac(x, y)
#Extendemos la matriz de X para el parámetro independiente
unos = np.array([np.ones(muestras)])
x = np.concatenate((unos.T, x), axis=1)
x = x.reshape(muestras, caracterisitcas+1)
y = y.reshape(np.size(y), 1)


# recordando lo aprendido anteriormente, dividimos nuestro cojunto de datos y normalizamos.

# In[ ]:


#Dejamos algunas muestras para el proceso de entrenamiento y otras para evaluar qué tan bueno fue el aprendizaje del modelo
random.seed(1)
N = muestras
ind=np.random.permutation(N)
Xtrain = x[ind[0:int(math.ceil(0.7*N))],:]
Xtest = x[ind[int(math.ceil(0.7*N)):N],:]
Ytrain = y[ind[0:int(math.ceil(0.7*N))]]
Ytest = y[ind[int(math.ceil(0.7*N)):N]]
# normalizamos
Xtrain, Xtest = normalizar(Xtrain, Xtest)


# Ahora vamos a completar el código de la regla de actualización de los parámetros del algoritmo de <font color='blue'>gradiente_descedente</font>: 
# 
# 
# $$w_j(iter) = w_j(iter-1) - \eta \frac{\partial E(w)}{\partial w_j}$$ 
# 
# recordar que 
# 
# $$ \frac{\partial E(w)}{\partial w_j} = \frac{\partial E({\bf{w}})}{\partial w_j} = \frac{1}{N}\sum_{i=1}^{N}\left( f({\bf{x}}_i,{\bf{w}}) - y_i\right) \frac{\partial }{\partial w_j} f({\bf{x}}_i, {\bf{w}})$$
# 
# Recuerda que debe usar las funciones ya implementadas y no usar **ninguna otra libreria**, adicional a las librerias ya pre-cargadas como numpy.
# 
# Adicionalmente, dentro de nuestra función, **vamos incluir una transformación polinómica**.
# 
# 
# Nota: Para el problema de clasificación tenga presente que si ya implementó la regla de actualización de parámetros para el modelo de regresión polinomial múltiple, este punto es trivial, puesto que sólo tiene que incluir la función sigmoidal tal como lo vimos en la teoría.

# In[ ]:


#ejercicio de codigo
def gradiente_descendente_logistic_poly(X,Y,grado,eta, iteraciones):
    """Gradiente descendente para regresión lineal múltiple
    X: Matriz de datos extendida
    Y: vector con los valores a predecir
    W: Vector de parámetros del modelo
    eta: Taza de aprendizaje
    grado: grado para usar en la transformacion polinomica
    iteraciones: numero de iteraciones maxima

    retorna: W el valor de de los parametros de regresión polinomica
    """
    X2 = potenciaPolinomio(X,grado)
    #Tomamos el número de variables del problema leugo de la transformacion
    d = np.size(X2,1)
    #Tomamos el número de muestras de la base de datos
    N = np.size(X2,0)   
    #Inicializamos el vector de parámetros
    W = np.zeros(d)
    W = W.reshape(np.size(W),1)
   
   
    for iter in range(iteraciones):
       
        #Aquí debe completar el código con la regla de actualización de los parámetros W para regresión
        #logística. Tenga en cuenta los nombres de las variables ya creadas: W, X, Y
        Y_estimado = logistic_regression(X2,W)

        W = 


    #Error en clasificación  
    Y_estimado = logistic_regression(X2,W)
    error_clasificacion = error_logistic(Y_estimado,Y)
    print("error despues de finalizar la iteraciones", error_clasificacion)
    return W


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio4", gradiente_descendente_logistic_poly)


# Finalmente se sugiere la siguiente funcion para evaluar el error del modelo. Entienda su funcionamiento.

# In[ ]:


def evaluar_modelo (W, X_to_test, Y_True, grado):
    """ funcion que evalua un modelo de regresión usando el error cuadratico medio

    W: es un matriz con los parametros del modelo entrenados
    X_to_test: conjunto de datos para usar en el evaluamiento del modelo
    Y_True: valores reales para usar en el evaluamiento del modelo
    grado: valor del polinomio a usar

    retorna: el de clasificación.
    """
    X2 = potenciaPolinomio(X_to_test,grado)
    Y_estimado = logistic_regression(X2,W)
    error_clasificacion = error_logistic(Y_estimado,Y_True)
    return(error_clasificacion)


# ## Ejercicio 3: Experimentar
# 
# Con la función implementada vamos a entrenar un modelo y calcular su error de entrenamiento

# In[ ]:


w = gradiente_descendente_logistic_poly(Xtrain,Ytrain,1,0.0001, 1000)
error_test = evaluar_modelo(w, Xtest, Ytest, grado = 1)
print("error en el conjunto de pruebas", error_test)


# En nuestro primer experimento vamos a evaluar el rendimiento del modelo usando varias tasas de aprendizaje y grados de polinimios. Vamos a dejar por ahora un numero de iteraciones fijas = 50. Para ello completa la siguiente función.

# In[ ]:


## ejercicio de codigo
def experimentar (Xtrain, Xtest, Ytrain, Ytest, tasas, grados):
    """ funcion para realizar experimentos.
    Xtrain: conjunto de datos
    Xtest:
    Ytrain:
    Ytest:
    tasas: Es una lista con los valores númericos de tasas de aprendizaje 
        para realizar los experimentos
    grados: Es una lista con los valores númericos de grados 
        para realizar los experimentos
    retorna: un dataframe con el resultados de los experimentos
    """
    numero_iter = 50

    resultados = pd.DataFrame()
    idx = 0 # indice
    for eta in tasas:
        for grado in grados:
            W = 
            error_entrenamiento = 
            error_prueba = 
            resultados.loc[idx,'grado'] = grado
            resultados.loc[idx,'tasa de aprendizaje'] = eta
            resultados.loc[idx,'error_entreamiento'] = error_entrenamiento
            resultados.loc[idx,'error_prueba'] = error_prueba
            idx = idx+1

    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio5", experimentar)


# In[ ]:


tasas = [1,0.1,0.001]
grados = [1,2,3,4,5]
resultados = experimentar (Xtrain, Xtest, Ytrain, Ytest, tasas, grados)


# In[ ]:


# para ver los resultados
resultados


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿con base a los resultados anteriores, qué efecto tiene el grado en los errores de entrenamiento y de prueba? justifique
respuesta_2 = "" #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿es normal que los errores de entrenamiento y prueba sean diferentes? justifique
respuesta_3 = "" #@param {type:"string"}


# En los problemas de clasificación, es muy importante entender el numero de muestras que el modelo clasifico mal. Esto es necesario observarlo tanto en entrenamiento como en pruebas. Vamos crear una función para esto.

# In[ ]:


#ejercicio de codigo
def numero_de_errores (W, X, Y, grado):
    """función que cuenta los errores de un modelo de regresión logistica
    W: vector con los parametros de un modelo de regresión logistica
       previamente entrenado
    X: conjunto de datos a usar (numpy matrix)
    Y: conjunto con las etiquetas verdaderas. (numpy array)
    grado: grado usado en el modelo de regresión logistica
    retorna: numero de errores (int/float) 
            (es decir el numero de veces que la etiqueta predicha es diferente a la etiqueta real)
    """
    X2 = potenciaPolinomio(X,grado)
    Y_estimado = logistic_regression(X2,W)
    numero_errores = np.sum(Y_estimado!=Y)

    return numero_errores


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio6", numero_de_errores)


# Ahora, use la función. Entrene nuevamente un modelo con los mejores parametros obtenidos y calcule cuantas muestras quedaron mal clasificadas. Tanto en el conjunto de entrenamiento y de prueba. 
# 
# Si hay parametros empatados, el modelo que tenga menos parametros deberia ser el mejor.
# 
# 

# In[ ]:


# puedes usar el siguiente código para ordenar los resultados y ver los 3 primeros
# resultados, usa esta salida, para ver cuales fueron los mejores parámetros
resultados.sort_values(by = ['error_prueba', 'grado'], ascending = True).head(3)


# In[ ]:


W = gradiente_descendente_logistic_poly(Xtrain,Ytrain,grado =  ,eta = , iteraciones = 50)
print("estos son los pesos para el modelo entrenando \n", W)


# In[ ]:


num_errores_entrenamiento =  numero_de_errores (W, Xtrain, Ytrain, grado = 4)
num_errores_prueba =  numero_de_errores (W, Xtest, Ytest, grado = 4)
print("muestras mal clasificadas en entrenamiento", num_errores_entrenamiento)
print("muestras mal clasificadas en pruebas", num_errores_prueba)


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿por qué se uso el error de prueba para ordenar la tabla de resultados en lugar del error de entrenamiento?
respuesta_4 = "" #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown Escribe aqui el numero de muestras que quedaron mal clasificadas en el conjunto de entrenamiento y de pruebas. ¿como calificarias el modelo entrenado?
respuesta_5 = "" #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown Escriba el modelo completo con sus variables y coeficientes de f(**x**,**w**) con la mejor frontera de decisión que encontró. usa los valores del último W entrenado. Recuerda tener presente el grado del polinomio
respuesta_6 = "0.0x1 + 1.0x1**2 + 2.0x2 + 3.0x2**3" #@param {type:"string"}


# In[ ]:


GRADER.check_tests()


# In[ ]:


#@title Integrantes
codigo_integrante_1 ='' #@param {type:"string"}
codigo_integrante_2 = ''  #@param {type:"string"}


# ----
# esta linea de codigo va fallar, es de uso exclusivo del los profesores
# 

# In[ ]:


GRADER.grade()

