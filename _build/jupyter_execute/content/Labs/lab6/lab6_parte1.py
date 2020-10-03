#!/usr/bin/env python
# coding: utf-8

# # Laboratorio 6 Parte 1
# 
# ### Reducción de dimensión: Selección de características
# 
# ### 2019-II
# 
# #### Profesor: Julián D. Arias Londoño
# #### julian.ariasl@udea.edu.co

# ### Primer integrante:
# Nombre:
# 
# 
# #### Segundo integrante:
# 
# Nombre:
# 

# ## Guía del laboratorio
# 
# En esta archivo va a encontrar tanto celdas de código cómo celdas de texto con las instrucciones para desarrollar el laboratorio.
# 
# Lea atentamente las instrucciones entregadas en las celdas de texto correspondientes y proceda con la solución de las preguntas planteadas.
# 
# Nota: no olvide ir ejecutando las celdas de código de arriba hacia abajo para que no tenga errores de importación de librerías o por falta de definición de variables.

# ## Indicaciones
# 
# Este ejercicio tiene como objetivo implementar varias técnicas de selección de características y usar SVM para resolver un problema de clasificación multiclase.
# 
# 

# Antes de iniciar a ejecutar las celdas, debe instalar la librería mlxtend que usaremos para los laboratorios de reducción de dimensión.
# Para hacerlo solo tiene que usar el siguiente comando: 
# `!pip install mlxtend`
# También puede consultar la guía oficial de instalación
#     de esta librería: https://rasbt.github.io/mlxtend/installation/
# 
# 
# 

# In[ ]:


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import time


# Para el problema de clasificación usaremos la siguiente base de datos: https://archive.ics.uci.edu/ml/datasets/Cardiotocography
# 
# 
# Analice la base de datos, sus características, su variable de salida y el contexto del problema.

# In[ ]:


#cargamos la bd de entrenamiento
db = np.loadtxt('DB/DB_Fetal_Cardiotocograms.txt',delimiter='\t')  # Assuming tab-delimiter

X = db[:,0:22]

#Solo para dar formato a algunas variables
for i in range(1,7):
    X[:,i] = X[:,i]*1000

X = X
Y = db[:,22]

print('Dimensiones de la base de datos de entrenamiento. dim de X: ' + str(np.shape(X)) + '\tdim de Y: ' + str(np.shape(Y)))


# ## Ejercicio 1: Entrenamiento sin selección de características

# En la siguiente celda de código no tiene que completar nada. Analice, comprenda y ejecute el código y tenga en cuenta los resultados para completar la tabla que se le pide más abajo.

# In[ ]:


def classification_error(y_est, y_real):
    err = 0
    for y_e, y_r in zip(y_est, y_real):

        if y_e != y_r:
            err += 1

    return err/np.size(y_est)

#Para calcular el costo computacional
tiempo_i = time.time()

#Creamos el clasificador SVM. Tenga en cuenta que el problema es multiclase. 
clf = svm.SVC(decision_function_shape='ovr', kernel='rbf', C = 100, gamma=0.0001)

#Implemetamos la metodología de validación

Errores = np.ones(10)
j = 0
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]  

    #Aquí se entran y se valida el modelo sin hacer selección de características
    
    ######
    
    # Entrenamiento el modelo.
    model = clf.fit(X_train,y_train)

    # Validación del modelo
    ypred = model.predict(X_test)
    
    #######

    Errores[j] = classification_error(ypred, y_test)
    j+=1

print("\nError de validación sin aplicar SFS: " + str(np.mean(Errores)) + " +/- " + str(np.std(Errores)))

print(('\n\nTiempo total de ejecución: ' + str(time.time()-tiempo_i)) + ' segundos.')


# 
# 
# 1.1 Describa la metodología de validación que se está aplicando.
# 
# R/:
# 

#     
# 1.2 ¿Con qué modelo se está resolviendo el problema planteado? ¿Cuáles son los parámetros establecidos para el modelo?
# 
# R/:

# ## Ejercicio 2: Entrenamiento con selección de características
# 
# En la siguiente celda, complete el código donde le sea indicado. Consulte la documentación oficial de la librería mlxtend para los métodos de selección de características. https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#sequential-feature-selector

# In[ ]:


'''
Recibe 4 parámetros: 
1. el modelo (clf para nuestro caso), 
2. el número de características final que se quiere alcanzar
3. Si es forward (True), si es Backward False,
4. Si es es flotante (True), sino False
'''
def select_features(modelo, n_features, fwd, fltg):

    
    sfs = SFS(modelo, 
           k_features=n_features, 
           forward=fwd,
           floating=fltg,
           verbose=1,
           scoring='accuracy',
           cv=0)
    
    return sfs


#Para calcular el costo computacional
tiempo_i = time.time()

#Implemetamos la metodología de validación 

Errores = np.ones(10)
j = 0
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]  
   
    #Aquí se entrena y se valida el modelo haciendo selección de características con diferentes estrategias
    
    #Complete el código llamando el método select_features con los parámetros correspondientes para responder el
    #Ejercicio 3.1
    sf = #Complete el código aquí

    #Complete el código para entrenar el modelo con las características seleccionadas. Tenga en cuenta
    #la metodología de validación aplicada para que pase las muestras de entrenamiento correctamente.
    sf = #Complete el código aquí
    
    Errores[j] = 1-sf.k_score_
    j+=1

print("\nError de validación aplicando SFS: " + str(np.mean(Errores)) + " +/- " + str(np.std(Errores)))

print("\nEficiencia en validación aplicando SFS: " + str(sf.k_score_*100) + "%" )

print ("\n\nTiempo total de ejecución: " + str(time.time()-tiempo_i) + " segundos.")


# ## Ejercicio 3
# 
# 3.1 En la celda de código anterior, varíe los parámetros correspondientes al número de características a seleccionar (use 3, 7 y 10) y la estrategia a implementar (SFS, SBS, SFFS, SBFS), para que complete la siguiente tabla de resultados:
# 

# In[3]:


import pandas as pd
import qgrid
df_types = pd.DataFrame({
    'Tecnica' : pd.Series(['SVM sin selección','SVM + SFS','SVM + SFS','SVM + SFS','SVM + SBS','SVM + SBS','SVM + SBS','SVM + SFFS','SVM + SFFS','SVM + SFFS','SVM + SBFS','SVM + SBFS','SVM + SBFS']),
    '# de características seleccionadas' : pd.Series([22,3,7,10,3,7,10,3,7,10,3,7,10]),
   })
df_types["Error de validación"] = ""
df_types["IC(std)"] = ""
df_types["Eficiencia"] = ""
df_types["Tiempo de ejecución"] = ""

df_types.set_index(['Tecnica','# de características seleccionadas'], inplace=True)
df_types["Error de validación"][8] = "0.019"
df_types["IC(std)"][8] = "0.002"
df_types["Tiempo de ejecución"][8] = "107.9 s"

#df_types.sort_index(inplace=True)
qgrid_widget = qgrid.show_grid(df_types, show_toolbar=False)
qgrid_widget


# In[4]:


qgrid_widget.get_changed_df()


# 3.2 Según la teoría vista en el curso, se está usando una función tipo filtro o tipo wrapper y cuál es?
# 
# R/:
# 
# 3.3 Con los resultados de la tabla anterior haga un análisis de cuál es el mejor resultado teniendo en cuenta tanto la eficiencia en la clasificación como el costo computacional del modelo y la estrategia implementada.
# 
# R/:

# 3.4 Haga uso del atributo sf.k_feature\_idx\_ (deje evidencia del código usado para esto) para identificar cuáles fueron las características seleccionadas en el mejor de los resultados encontrados. No presente los indices de las características sino sus nombres y descripción.
# 
# R/: 
# 
# 3.5 De acuerdo a los resultados encontrados y la respuesta anterior, usted como ingeniero de datos que le puede sugerir a un médico que esté trabajando en un caso enmarcado dentro del contexto de la base de datos trabajada, para que apoye su diagnóstico?
# 
# R/: 
