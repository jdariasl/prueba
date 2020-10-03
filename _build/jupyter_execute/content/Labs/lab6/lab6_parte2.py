#!/usr/bin/env python
# coding: utf-8

# # Laboratorio 6 Parte 2
# 
# ### Reducción de dimensión: PCA y LDA
# 
# ### 2019-II
# 
# #### Profesor: Julián D. Arias Londoño
# #### julian.ariasl@udea.edu.co

# ## Estudiantes
# 
# #### Primer Integrante:
# #### Segundo Integrante:

# ## Guía del laboratorio
# 
# En esta archivo va a encontrar tanto celdas de código cómo celdas de texto con las instrucciones para desarrollar el laboratorio.
# 
# Lea atentamente las instrucciones entregadas en las celdas de texto correspondientes y proceda con la solución de las preguntas planteadas.
# 
# Nota: no olvide ir ejecutando las celdas de código de arriba hacia abajo para que no tenga errores de importación de librerías o por falta de definición de variables.

# ## Indicaciones
# 
# Este ejercicio tiene como objetivo implementar varias técnicas de extracción de características (PCA y LDA) y usar SVM para resolver un problema de clasificación multietiqueta o multiclase.
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
# 

# In[ ]:


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from mlxtend.feature_extraction import LinearDiscriminantAnalysis as LDA

import time


# 
# Para el problema de clasificación usaremos la siguiente base de datos: https://archive.ics.uci.edu/ml/datasets/Cardiotocography
# 
# 
# 
# Analice la base de datos, sus características, su variable de salida y el contexto del problema.

# 
# 
# Analice y comprenda la siguiente celda de código donde se importan las librerías a usar y se carga la base de datos.

# In[ ]:


#cargamos la bd de entrenamiento
db = np.loadtxt('DB/DB_Fetal_Cardiotocograms.txt',delimiter='\t')  # Assuming tab-delimiter

X = db[:,0:22]

#Solo para dar formato a algunas variables
for i in range(1,7):
    X[:,i] = X[:,i]*1000

X = X
Y = db[:,22]

#Para darle formato de entero a la variable de salida

Y_l = []
for i in Y:
    Y_l.append(int(i))
Y = np.asarray(Y_l)

print ("Dimensiones de la base de datos de entrenamiento. dim de X: " + str(np.shape(X)) + "\tdim de Y: " + str(np.shape(Y)))


# ## Ejercicio 1: Entrenamiento sin extracción de características
# 
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

print("\nError de validación sin aplicar extracción: " + str(np.mean(Errores)) + " +/- " + str(np.std(Errores)))

print ("\n\nTiempo total de ejecución: " + str(time.time()-tiempo_i) + " segundos.")

#print str(ypred)
#print str(y_test) 


# 
# 
# 1.1 Cuando se aplica PCA ¿es necesario estandarizar los datos? Si, No y por qué? En qué consiste dicha estandarización?
# 
# R/:
#     
# 1.2 La proyección de los datos que realiza PCA busca optimizar un medida, ¿Cuál? Explique.
# 
# R/:

# ## Ejercicio 2: Entrenamiento con extracción de características
# 
# En la siguiente celda, complete el código donde le sea indicado. Consulte la documentación oficial de la librería mlxtend para los métodos de extracción de características. https://rasbt.github.io/mlxtend/user_guide/feature_extraction/

# In[1]:


'''
Feature Extraction Function
#Recibe 2 parámetros: 
1. el tipo de método de extracción (pca o lda como string),
2. el número componentes (para pca) o el número de discriminantes (para lda)

#Para este laboratorio solo se le pedirá trabajar con PCA, LDA es opcional.
'''

def extract_features(tipo, n):
    
    if tipo == 'pca':
    
        ext = PCA(n_components=n)
    
        return ext

    elif tipo == 'lda':
        
        ext = LDA(n_discriminants=n)
        
        return ext
    
    else:
        print ("Ingrese un método válido (pca o lda)\n")


# In[ ]:



#Para calcular el costo computacional
tiempo_i = time.time()

#Estandarizamos los datos
X = standardize(X)

#Implemetamos la metodología de validación cross validation con 10 folds

Errores = np.ones(10)
j = 0
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X):
    
    #Aquí se aplica la extracción de características por PCA
    #Complete el código
    
    ex = #Complete el código llamando el método extract_features. Tenga en cuenta lo que le pide el ejercicio 3.1

    #Fit de PCA
    ex = #Complete el código con el fit correspondiente
    
    #Transforme las variables y genere el nuevo espacio de características de menor dimensión
    X_ex = #complete el código aquí para hacer la transformación
    
    
    #Aquí se aplica la extracción de características por LDA
    
    #OPCIONAL
    '''
    ex = #Complete el código llamando el método extract_features.Tenga en cuenta lo que le pide el ejercicio 3.1

    #Fit de LDA
    ex = #Complete el código con el fit correspondiente
    
    #Transforme las variables y genere el nuevo espacio de características de menor dimensión
    X_ex = #complete el código aquí para hacer la transformación
    '''
    
    #Se aplica CV-10
    
    X_train, X_test = X_ex[train_index], X_ex[test_index]
    y_train, y_test = Y[train_index], Y[test_index]  
   
    #Aquí se entrena y se valida el modelo luego de aplicar extracción de características con PCA o LDA
    
    ######
    
    # Entrenamiento el modelo.
    model = clf.fit(X_train,y_train)

    # Validación del modelo
    ypred = model.predict(X_test)
    
    #######

    Errores[j] = classification_error(ypred, y_test)
    j+=1
        

print("\nError de validación aplicando extracción: " + str(np.mean(Errores)) + " +/- " + str(np.std(Errores)))

print("\nEficiencia en validación aplicando extracción: " + str((1-np.mean(Errores))*100) + "%" )

print ("\n\nTiempo total de ejecución: " + str(time.time()-tiempo_i) + " segundos.")


# ## Ejercicio 3
# 
# 3.1 En la celda de código anterior, varíe los parámetros correspondientes al número de componentes principales a tener en cuenta (use 2, 10, 19 y 21 componentes principales) para PCA y complete la siguiente tabla de resultados:

# In[2]:


import pandas as pd
import qgrid
df_types = pd.DataFrame({
    'Tecnica' : pd.Series(['SVM sin extracción','SVM + PCA','SVM + PCA','SVM + PCA','SVM + PCA']),
    '# de características seleccionadas' : pd.Series(['N/A',2,10,19,21]),
   })
df_types["Error de validación"] = ""
df_types["IC(std)"] = ""
df_types["Eficiencia"] = ""
df_types["Tiempo de ejecución"] = ""

df_types.set_index(['Tecnica','# de características seleccionadas'], inplace=True)

#df_types.sort_index(inplace=True)
qgrid_widget = qgrid.show_grid(df_types, show_toolbar=False)
qgrid_widget


# In[4]:


qgrid_widget.get_changed_df()


# 3.2 Analizando los resultados del punto anterior que puede decir de la viabilidad de aplicar PCA para hacer reducción de dimensión en este problema?
# 
# R/:
# 

# 3.3 Explique en sus palabras la principal ventaja que tiene LDA sobre PCA para resolver problemas de clasificación.
# 
# R/: 

# 3.3 Explique en sus palabras las diferencias que existen entre los métodos de selección de características y los métodos de extracción de características vistos en el curso.
# 
# R/: 
