#!/usr/bin/env python
# coding: utf-8

# # Laboratorio 5 - Parte 1
# 
# ### Máquinas de Vectores de Soporte
# 
# ### 2019-II
# 
# #### Profesor: Julián D. Arias Londoño
# #### julian.ariasl@udea.edu.co
# 

# ## Guía del laboratorio
# 
# En esta archivo va a encontrar tanto celdas de código cómo celdas de texto con las instrucciones para desarrollar el laboratorio.
# 
# Lea atentamente las instrucciones entregadas en las celdas de texto correspondientes y proceda con la solución de las preguntas planteadas.
# 
# Nota: no olvide ir ejecutando las celdas de código de arriba hacia abajo para que no tenga errores de importación de librerías o por falta de definición de variables.

# #### Primer Integrante:
# #### Segundo Integrante:

# In[2]:


from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
#Algunas advertencias que queremos evitar
import warnings
warnings.filterwarnings("always")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Ejercicio 1: Completar código y se usa la estrategia 'ovr' por defecto de la librería SVC
# 
# En este ejercicio deben resolver un problema de clasificación multi-clase usando una SVM. Teniendo en cuenta que la formulación original de la SVM sólo permite resolver problemas bi-clase, deben comparar los resultados obtenidos usando una estrategia **Uno vs Uno** con una estrategia **Uno vs El resto**.

# Cargamos los datos:

# In[10]:


from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
digits = load_digits(n_class=4)

#--------- preprocesamiento--------------------
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)

#---------- Datos a usar ----------------------
X = data
Y = digits.target


# Consutar el manejo de la librería sklearn para entrenar un modelos SVM en: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC. Para el caso multiclase la librería ya tiene implementada la estrategia **Uno vs Uno**, así que en este caso sólo deben llamar correctamente los métodos. Complete el siguiente código usando un clasificador basado en SVM:

# In[ ]:


import math
import numpy as np
from numpy import random
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


#Validamos el modelo
Folds = 4
random.seed(19680801)
EficienciaTrain = np.zeros(Folds)
EficienciaVal = np.zeros(Folds)
skf = StratifiedKFold(n_splits=Folds)
j = 0
for train, test in skf.split(X, Y):
    Xtrain = X[train,:]
    Ytrain = Y[train]
    Xtest = X[test,:]
    Ytest = Y[test]
    
    #Normalizamos los datos
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    
    #Haga el llamado a la función para crear y entrenar el modelo usando los datos de entrenamiento
    modelo = ....
    
    
    #Validación
    Ytrain_pred = ...
    Yest = ...
    
    #Evaluamos las predicciones del modelo con los datos de test
    EficienciaTrain[j] = np.mean(Ytrain_pred.ravel() == Ytrain.ravel())
    EficienciaVal[j] = np.mean(Yest.ravel() == Ytest.ravel())
    j += 1
        
print('Eficiencia durante el entrenamiento = ' + str(np.mean(EficienciaTrain)) + '+-' + str(np.std(EficienciaTrain)))
print('Eficiencia durante la validación = ' + str(np.mean(EficienciaVal)) + '+-' + str(np.std(EficienciaVal)))


# ## Ejercicio 2: Experimentos

# Realice los experimientos necesarios para llenar la siguiente tabla:

# In[22]:


import pandas as pd
import qgrid
df_types = pd.DataFrame({
    'Kernel' : pd.Series(['lineal','lineal','lineal','lineal','lineal','lineal','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf']),
    'C' : pd.Series([0.001,0.01,0.1,1,10,100,0.001,0.001,0.001,0.01,0.01,0.01,0.1,0.1,0.1,1,1,1,10,10,10,100,100,100]),
    'gamma' : pd.Series([0,0,0,0,0,0,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1])})
df_types["Eficiencia en validacion"] = ""
df_types["Intervalo de confianza"] = ""
df_types["% de Vectores de Soporte"] = ""
df_types.set_index(['Kernel','C','gamma'], inplace=True)
df_types["Eficiencia en validacion"][3] = "0.97077"
df_types["Intervalo de confianza"][3] = "0.01548"
df_types["% de Vectores de Soporte"][3] = "0.2620"
#df_types.sort_index(inplace=True)
qgrid_widget = qgrid.show_grid(df_types, show_toolbar=False)
qgrid_widget


# Ejecute la siguiente instrucción para dejar guardados en el notebook los resultados de las pruebas.

# In[23]:


qgrid_widget.get_changed_df()


# ## Ejercicio 3: Completar código sin utilizando la estrategia 'ovr' por defecto de la librería SVC
# 
# Cree dos funciones, una para entrenar un conjunto de modelos bajo la estrategia Uno vs el resto, usando como clasificador base una SVM. La segunda función debe usar el conjunto de modelos entrenados, y clasificar un conjunto de muestras de validación.
# 
# #### Nota: Completar la estrategia OVR a mano, no se permite el uso de alguna librería externa ni tampoco utilizar el parámetro "ovr" por defecto de la librería SVC.
# 

# In[ ]:


def TrainSVM_OnevsRest():
    
    

def ValidaSVM_OnevsRest():
    


# ## Ejercicio 4: Entrenamiento

# Use las funciones definidas en el punto anterior para llevar a cabo la prueba de simulación con el mismo conjunto de datos del punto 1.

# In[ ]:


Folds = 4
random.seed(19680801)
EficienciaTrain = np.zeros(Folds)
EficienciaVal = np.zeros(Folds)
skf = StratifiedKFold(n_splits=Folds)
j = 0
for train, test in skf.split(X, Y):
    Xtrain = X[train,:]
    Ytrain = Y[train]
    Xtest = X[test,:]
    Ytest = Y[test]
    
    #Normalizamos los datos
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    
    #Haga el llamado a la función para crear y entrenar el modelo usando los datos de entrenamiento
    
    modelo = ...
    
    
    #Validación
    Ytrain_pred = ...
    Yest = ...
    
    #Evaluamos las predicciones del modelo con los datos de test
    EficienciaTrain[j] = np.mean(Ytrain_pred.ravel() == Ytrain.ravel())
    EficienciaVal[j] = np.mean(Yest.ravel() == Ytest.ravel())
    j += 1
        
print('Eficiencia durante el entrenamiento = ' + str(np.mean(EficienciaTrain)) + '+-' + str(np.std(EficienciaTrain)))
print('Eficiencia durante la validación = ' + str(np.mean(EficienciaVal)) + '+-' + str(np.std(EficienciaVal)))


# ## Ejercicio 5: Experimentos

# Realice los experimientos necesarios para llenar la siguiente tabla:

# In[35]:


import pandas as pd
import qgrid
df_types = pd.DataFrame({
    'Kernel' : pd.Series(['lineal','lineal','lineal','lineal','lineal','lineal','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf','rbf']),
    'C' : pd.Series([0.001,0.01,0.1,1,10,100,0.001,0.001,0.001,0.01,0.01,0.01,0.1,0.1,0.1,1,1,1,10,10,10,100,100,100]),
    'gamma' : pd.Series([0,0,0,0,0,0,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1,0.01,0.1,1])})
df_types["Eficiencia en validación"] = ""
df_types["Intervalo de confianza"] = ""
df_types["% de Vectores de Soporte"] = ""
df_types.set_index(['Kernel','C','gamma'], inplace=True)
df_types["Eficiencia en validacion"][3] = "0.97633"
df_types["Intervalo de confianza"][3] = "0.01837"
df_types["% de Vectores de Soporte"][3] = "0.2778"
#df_types.sort_index(inplace=True)
qgrid_widget = qgrid.show_grid(df_types, show_toolbar=False)
qgrid_widget


# Ejecute la siguiente instrucción para dejar guardados en el notebook los resultados de las pruebas.

# In[36]:


qgrid_widget.get_changed_df()


# *En las tablas el punto separa las cifras decimales, es decir 100.000 es 100 ó 1.000 es 1.

# In[ ]:




