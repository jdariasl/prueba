#!/usr/bin/env python
# coding: utf-8

# # Laboratorio 5 - Parte 2
# 
# ### Máquinas de Vectores de Soporte
# 
# ### 2019-II
# 
# #### Profesor: Julián D. Arias Londoño
# #### julian.ariasl@udea.edu.co

# ## Guía del laboratorio
# 
# En esta archivo va a encontrar tanto celdas de código cómo celdas de texto con las instrucciones para desarrollar el laboratorio.
# 
# Lea atentamente las instrucciones entregadas en las celdas de texto correspondientes y proceda con la solución de las preguntas planteadas.
# 
# Nota: no olvide ir ejecutando las celdas de código de arriba hacia abajo para que no tenga errores de importación de librerías o por falta de definición de variables.

# #### Primer Integrante:
# #### Segundo Integrante:

# In[16]:


from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
#Algunas advertencias que queremos evitar
import warnings
warnings.filterwarnings("always")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Ejercicio 1: Limipiar base de datos y completar código
# 
# En este ejercicio usaremos la regresión por vectores de soporte para resolver el problema de regresión de la base de datos AirQuality (https://archive.ics.uci.edu/ml/datasets/Air+Quality).

# ### Limipiar base de datos
# La siguiente celda de código limpia la base de datos de todos sus datos faltantes y la deja lista en la variable DataBase.

# 1. **Cargar** la base de datos
# 2. **Quitar** todos registros de la base de datos que son perdidos y están marcados como -200, es decir, donde haya un valor -200 eliminaremos el registro.
# 3.  Ya hemos eliminado los registros con valor de la variable de salida perdido. Ahora vamos a **imputar los valores perdidos** en cada una de las características.
# 4. **Verificar** si quedaron valores faltante

# In[17]:


#Paso 1: Cargar
db = np.loadtxt('BDatos/AirQuality.data',delimiter='\t')  # Assuming tab-delimiter
print("Dim de la base de datos original: " + str(np.shape(db)))
db = db.reshape(9357,13)

DataBase = db

#Paso 2: Quitar
j = 0
for i in range(0,np.size(db,0)):
    if -200 == db[i,12]:
        #print i
        j+=1
        DataBase = np.delete(DataBase,i,0)
    
print ("\nHay " + str(j) + " valores perdidos en la variable de salida.")

print ("\nDim de la base de datos sin las muestras con variable de salida perdido "+ str(np.shape(DataBase)))


##Paso 3: Imputar
print ("\nProcesando imputación de valores perdidos en las características . . .\n")


for k in range(0,np.size(DataBase,0)):
    for w in range(0,13):
        if -200 == DataBase[k,w]:
            DataBase[k,w] = round(np.mean(DataBase[:,w])) ## Se imputa con la media de toda la caracteristicas
        
print ("Imputación finalizada.\n")


##Paso 4: Verificar

hay_missed_values = False
for i in range(0,np.size(DataBase,0)):
    if -200 in DataBase[i,:]:
        hay_missed_values = True
if(hay_missed_values):
    print ("Hay valores perdidos")
else:
    print ("No hay valores perdidos en la base de datos. Ahora se puede procesar. La base de datos está en la variable DataBase")


# Base de datos final

# In[14]:


X = DataBase[:,0:12]

Y = DataBase[:,12]


# Definimos la función Mean Absolute Percentage Error para los problemas de regresión

# In[13]:


def MAPE(Y_est,Y):
    ind = Y != 0 #Remueve los elementos que son cero en la variable deseada
    N = np.size(Y[ind])
    mape = np.sum(abs((Y_est[ind].reshape(N,1) - Y[ind].reshape(N,1))/(Y[ind].reshape(N,1)+np.finfo(np.float).eps)))/N
    return mape 


# 
# 
# ### Complete el código
# 
# A continuación complete el siguiente código para crear el modelo vectores de soporte(SVM) para regresión usando la librería sklearn. 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html. Implementar la metodología cross-validation con 5 folds.

# In[ ]:


from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.svm import SVR
import time
tiempo_i = time.time()

#Complete el código para crear el modelo SVM para regresión. 
#Use un kernel rbf con una malla de valores así: C en {0.1, 100} y gamma en {0.0001, 0.1}
#clf = ...


Folds = 
Errores = np.ones(Folds)
j = 0
kf = KFold(n_splits=Folds)

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]  
       
    #Complete el código
    
    
    
    # Entrenar el modelo
    #modelo = ...
    
    
    # Validación del modelo
    #ypred = ...
 
    Errores[j] = MAPE(ypred, y_test)
    j+=1
    
print("\nError de validación: " + str(np.mean(Errores)) + " +/- " + str(np.std(Errores)))

print ("\n\nTiempo total de ejecución: " + str(time.time()-tiempo_i) + " segundos.")


# ## Ejercicio 2: Experimentos
# 
# Una vez complete el código, realice las simulaciones necesarias para llenar la tabla siguiente:

# In[58]:


import pandas as pd
import qgrid
df_types = pd.DataFrame({
    'Kernel' : pd.Series(['lineal','lineal','lineal','lineal','lineal','lineal','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf']),
    'C' : pd.Series([0.001,0.01,0.1,1,10,100,0.001,0.001,0.001,0.01,0.01,0.01,0.1,0.1,0.1,1,1,1,10,10,10,100,100,100]),
    'gamma' : pd.Series([0,0,0,0,0,0,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1])})
df_types["MAPE Promedio"] = ""
df_types["Intervalo de confianza"] = ""
df_types["% de Vectores de Soporte"] = ""
df_types.set_index(['Kernel','C','gamma'], inplace=True)
df_types["MAPE Promedio"][23] = "0.2259"
df_types["Intervalo de confianza"][23] = "0.1109"
df_types["% de Vectores de Soporte"][23] = "0.2191"
#df_types.sort_index(inplace=True)
qgrid_widget = qgrid.show_grid(df_types, show_toolbar=False)
qgrid_widget


# Ejecute la siguiente instrucción para dejar guardados en el notebook los resultados de las pruebas.

# In[59]:


qgrid_widget.get_changed_df()


# ## Ejercicio 3: Completar preguntas

# 3.1 ¿Cuál es la finalidad de usar las funciones kernel en el modelo SVM?
# 
# R/:
# 

# 3.2 ¿En este caso el porcentaje de vectores de soporte provee una información similar que en el problema de clasificación? Explique su respuesta.
# 
# R/:
# 

# 3.3 Realice una gráfica de las salidas reales vs las predicciones del modelo SVM, para evaluar visualmente el desempeño del mismo. Esto solo para la configuración en la cuál se encontró el menor error.
# 
# Complete el código para hacer la gráfica aquí

# In[ ]:




