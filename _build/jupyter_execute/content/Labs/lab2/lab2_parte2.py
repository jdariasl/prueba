#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jdariasl/ML_2020/blob/master/Labs/lab2/lab2_parte2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# **Recuerda que una vez abierto, Da clic en "Copiar en Drive", de lo contrario no podras alamancenar tu progreso**
# 
# Nota: no olvide ir ejecutando las celdas de código de arriba hacia abajo para que no tenga errores de importación de librerías o por falta de definición de variables.

# In[ ]:


#configuración del laboratorio
# Ejecuta esta celda!
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# for local 
#import sys ; sys.path.append('../commons/utils/')
get_ipython().system('wget https://raw.githubusercontent.com/jdariasl/ML_2020/master/Labs/commons/utils/general.py -O general.py')
from general import configure_lab2
configure_lab2()
from lab2 import *
GRADER, x, y = part_2()


# # Laboratorio 2 - Parte 2
# 
# 

# ## Ejercicio 1: Contextualización del problema
# 
# Para el problema de regresion usaremos la base de datos 'The Boston Housing Dataset', cuya descripción [pueden encontrarla aqui](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html). La información ya esta cargada dentro del notebook

# In[ ]:


print("muestra de los 3 primeros renglones de x:\n", x[0:3, :])
print("muestra de los 3 primeros renglones de y:\n", y[0:3])
print ("¿el resultado de esta instrucción que información nos brinda?", x.shape[0])
print ("¿el resultado de esta instrucción que información nos brinda?", x.shape[1])
print ("¿el resultado de esta instrucción que información nos brinda?", len(np.unique(y)))


# En los problemas de regresión, es muy util explorar la distribución de la variable objetivo. Nuestro primer ejercicio consiste en:
# 1. visualizar un histograma de la variable y 
# 2. retornar los intervalo de datos mas frecuente.
# 
# Pistas: 
# 1. explorar la documentación de [plt.hist](https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.hist.html). Maneje los valores por defecto. ¿como se puede usar la salida del histograma para retorna el intervalo de datos mas frecuente?
# 2. ¿ `np.argsort(numpy_array)[::-1]` que efecto tiene?

# In[ ]:


#ejercicio de código
def plot_hist_and_get_freq_int(Y):
    """función que grafica el histograma de la variable 'Y'
        y retorna el intervalo donde ocurren con mas frecuencia los
        valores de 'Y'
        Y: numpy array con la variable a graficar
        retorna: una tupla (int/float, int/float, int/float) 
            el primer elemento es al limite inferior del intervalo donde ocurren los valores
            mas frecuentes
            el segundo elemento es al limite superior del intervalo donde ocurren los valores
            mas frecuentes
            el tercer elemento es el la frecuencia del intervalo
            va observar un cuarto elemento a retornar, el cual es usado para confirmar que
            se realizo la grafica correctamente
    """
    
    plt.hist()
    lim_inf = 
    lim_sup = 
    freqs =
    
    # el cuarto elemento debe dejarlo
    return (lim_inf, lim_sup, freqs, plt.gcf())


# In[ ]:


## la funcion que prueba tu implementacion
#ignora las graficas!!
GRADER.run_test("ejercicio1", plot_hist_and_get_freq_int)


# In[ ]:


# ver el histograma!
plot_hist_and_get_freq_int(y)


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿evaluando **solo** el histograma, podria decirse que nuestra variable 'y' podria modelarse de manera **totalmente exacta** con una sola distribución de probabilidad gausiana? justifique su respuesta
respuesta_1 = "" #@param {type:"string"}


# ## Ejercicio 2: Completar código de K-Vecinos para regresión.
# 
# Vamos a implementar ahora KNN para un problema de regresión.

# Las mismas pistas de nuestro laboratorio anterior son de utilidad para implementar el algoritmo.
# 
# 1. Para el cáculo de la distancia entre vectores existen varias opciones:
#     1. usar la función la distancia entre matrices `scipy.spatial.distance.cdist`([Ejemplo](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist))--esta puede ser usada directamente como `cdist(...)`. Entiende la salida de esta función. Al usarla, se logra un rendimiento superior.
#     2. usar la función la distancia euclidiana `scipy.spatial.distance.euclidean`([Ejemplo](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html))--pueder acceder a ella directamente como `euclidean`. Aca debe pensar en un algoritmo elemento a elemento, por lo tanto menos eficiente.
# 2. También serán de utilidad las funciones `np.sort` y `np.argsort`.
# 3. ¿cual es la unica diferencia entre el knn para clasificación y regresión? en lugar de la moda, que metodo debemos usar?

# In[ ]:


#ejercicio de código
def KNN_regresion(X_train, Y_train, X_test, k):
    """ Funcion que implementa el modelo de K-Vecino mas cercanos
        para regresión
    X_train: es la matriz con las muestras de entrenamiento
    Y_train: es un vector con los valores de salida pra cada una de las muestras de entrenamiento
    X_test: es la matriz con las muestras de validación
    k (int): valor de vecinos a usar
    retorna: las estimaciones del modelo KNN para el conjunto X_test 
             esta matriz debe tener un shape de [row/muestras de X_test] 
             y las distancias de X_test respecto a X_train, estan matrix
             debe tener un shape de [rows de X_test, rows X_train]
             lo que es lo mismo [muestras de X_test, muestras de X_train]
    """
    
    if k > X_train.shape[0]:
        print("k no puede ser menor que las muestras de entrenamiento")
        return(None)

    distancias =  
    Yest = np.zeros(X_test.shape[0])
  
        
    return Yest, distancias


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio2", KNN_regresion)


# ## Ejercicio 3: Experimentos con KNN
# 
# Ahora vamos a probar nuestro algoritmo. Antes de ello, definos la función para calcular el error

# In[ ]:


def MAPE(Y_est,Y):
    """Mean Absolute Percentage Error para los problemas de regresión
    Y_est: numpy array con los valores estimados
    Y: numpy array con las etiquetas verdaderas
    retorna: mape
    """
    N = np.size(Y)
    mape = np.sum(abs((Y_est.reshape(N,1) - Y.reshape(N,1))/Y.reshape(N,1)))/N
    return mape 


# Y ahora, si, vamos a crear la función para experimentar.
# 
# En el ejercicio de código, se puede observar que usamos nuevamente la funciónes de la libreria **sklearn**:
# 
# 1. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) para normalizar.
# 
# 2. [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Para dividir el conjunto de datos. Entiende como estamos usando esta función.

# In[ ]:


#@title Pregunta Abierta
#@markdown si bien, dentro del código es usada la función train_test_split, que metodologia de validación es implementada usando esta función? justifique
respuesta_2 = "" #@param {type:"string"}


# In[ ]:


#Ejercicio de código
def experimentar (X, Y, ks):
    """Función que realiza los experimentos con knn usando
       una estrategia de validacion entrenamiento y pruebas
    X: matriz de numpy conjunto con muestras y caracteristicas
    Y: vector de numpy con los valores a predecir
    ks: List[int/float] lista con los valores de k-vecinos a usar
    retorna: dataframe con los resultados, debe contener las siguientes columnas:
        - los k-vecinos, el error-mape medio de prueba, la desviacion estandar del error-mape
    """
    
    resultados = pd.DataFrame()
    idx = 0
    # iteramos sobre la lista de k's
    for k in ks:
        # lista para almacenar los errores de cada iteración
        # de la validación
        error_temp = []
        
        # iteramos para validar
        for j in range(3): 
            # dividimos usando la función
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y)
            scaler = StandardScaler()
            scaler.fit(Xtrain)
            Xtrain= scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)

            Yest, _ = KNN_regresion(...)
            errorTest =
            error_temp.append(errorTest)
    
        resultados.loc[idx,'k-vecinos'] = k 
        resultados.loc[idx,'error de prueba(media)'] =
        resultados.loc[idx,'error de prueba(desviación estandar)'] =
        idx+=1

    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio3", experimentar)


# Ahora ejecuta los experimentos con k = 2,3,4,5,6,7,10

# In[ ]:


resultados = experimentar (x, y,[2,3,4,5,6,7,10])
resultados


# ## Ejercicio 4: Ventana de Parzen y experimentos
# 
# Ahora, igualmente, vamos aplicar ventana de parzen para resolver el problema de regresión.
# 
# $$f({\bf{x}}^*) = \frac{1}{N h^d} \sum_{i=1}^{N} K(u_i), \;\; u_i = \frac{d({\bf{x}}^*,{\bf{x}}_i)}{h}$$
# 
# En la siguiente celda se define la función para un $K(u_i)$ gaussiano y se realiza la sugerencia para estimar el termino $ \sum_{i=1}^{N} K(u_i)$, siendo $\;\; u_i = \frac{d({\bf{x}}^*,{\bf{x}}_i)}{h}$. 
# 
# Observa y entiende esta última función y sus argumentos. Recordando que para regresión, debemos usar la relación de **Nadaraya_Watson**.
# 
# $$y^* = \frac{\sum_{i=1}^N K(u_i)y_i}{\sum_{i=1}^N K(u_i)} $$
# 
# 

# In[ ]:


def kernel_gaussiano(x):
    return (np.exp((-0.5)*x**2))

def ParzenWindow(x,Data,h,Datay=None):
    """"ventana de parzen
    x: vector con representando una sola muestra
    Data: vector de muestras de entrenamiento
    h: ancho de la ventana de kernel
    Datay: vector con los valores de salida (y), Si no se pasa como argumento, 
        se calcula un ventana de parzen sin multiplicar los valores de este vector.
    retorna: el valor de ventana de parzen para una muestra
    """
    h = h
    Ns = Data.shape[0]
    suma = 0
    for k in range(Ns):
        u = euclidean(x,Data[k,:])
        if Datay is None:
            suma += kernel_gaussiano(u/h)
        else:
            suma += kernel_gaussiano(u/h)*Datay[k]
    return suma


# Usando las anteriores funciones, completa el código.

# In[ ]:


#Ejercicio de código
def Nadaraya_Watson(X_train, Y_train, X_test, h):
    """ Funcion que implementa metodo de ventana de parzen para
        para clasificación
    X_train: es la matriz con las muestras de entrenamiento
    Y_train: es un vector con los valores de salida pra cada una de las muestras de entrenamiento
    X_test: es la matriz con las muestras de validación
    h (float): ancho de h de la ventana
    retorna: - las estimaciones del modelo parzen para el conjunto X_test 
              esta matriz debe tener un shape de [row/muestras de X_test]
             - las probabilidades de la vetana [row/muestras de X_test, numero de clases]  
    """
        
    Yest = np.zeros(X_test.shape[0])
    
    
    #Debe retornar un vector que contenga las predicciones para cada una de las muestras en X_val, en el mismo orden.  
    return Yest


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio4", Nadaraya_Watson)


# ## Ejercicio 5: Experimentos con Parzen
# En el ejercicio de código, se puede observar que usamos nuevamente la funciónes de la libreria **sklearn**:
# 
# 1. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) para normalizar.
# 2. Y se debe usar la función [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html?highlight=kfold#sklearn.model_selection.KFold) para realizar la validación. Tener en cuenta la documentación para poder completar el código de manera correcta.

# In[ ]:


#@title Pregunta Abierta
#@markdown ¿cual es la metodologia de validación usada en el experimento? ¿qué diferencia tiene respecto a la metodologia usada en el primer experimento?
respuesta_3 = "" #@param {type:"string"}


# In[ ]:


def experimentarParzen (X, Y, hs):
    """Función que realiza los experimentos con knn usando
       una estrategia de validacion entrenamiento y pruebas
    X: matriz de numpy conjunto con muestras y caracteristicas
    Y: vector de numpy con los valores de las etiquetas
    ks: List[int/float] lista con los valores de k-vecinos a usar
    retorna: dataframe con los resultados, debe contener las siguientes columnas:
        - el ancho de ventana, 
        - el error medio de prueba
        - la desviacion estandar del error
        - número de promedio en el conjunto de prueba/validacion
    """
    # se usa la función para implementar la estrategia de validación.
    kfolds = KFold(n_splits=4)
    resultados = pd.DataFrame()
    idx = 0
    # iteramos sobre los valores de hs
    for h in hs:
        # lista para almacenar los errores y numero de muestras
        # de cada iteración
        # de la validación
        error_temp = []
        numero_muestras = []
        
        for train, test in kfolds.split( ):

            Xtrain = X[,:]
            Ytrain = Y[]
            Xtest = X[,:]
            Ytest = Y[]
            #normalizamos los datos
            scaler = StandardScaler()
            scaler.fit(Xtrain)
            Xtrain = scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)
            
            Yest = Nadaraya_Watson(...)
            errorTest = 
            error_temp.append(errorTest)
            numero_muestras.append()
    
        resultados.loc[idx,'ancho de ventana'] = h 
        resultados.loc[idx,'error de prueba(media)'] = 
        resultados.loc[idx,'error de prueba(desviación estandar)'] =
        resultados.loc[idx,'muestras en conjunto de pruebas (media)'] = 
        idx+=1
    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio5", experimentarParzen)


# In[ ]:


# ejecute para ver los experimentos
hs = [1,1.5 ,2.5, 5, 10]
experimentos_parzen = experimentarParzen(x,y, hs)
experimentos_parzen


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿es normal que el la media de muestra en el cojunto de pruebas siempre es la misma? justifique
respuesta_4 = "" #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿que metodo podria usarse para estimar el valor de h?
respuesta_5 = "" #@param {type:"string"}


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

