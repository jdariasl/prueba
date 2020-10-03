#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Complejidad de modelos </font>

# In[1]:


get_ipython().system('wget --no-cache -O init.py -q https://raw.githubusercontent.com/jdariasl/ML_2020/master/init.py')
import init; init.init(force_download=False); 


# ### Julián D. Arias Londoño
# 
# Profesor Asociado  
# Departamento de Ingeniería de Sistemas  
# Universidad de Antioquia, Medellín, Colombia  
# julian.ariasl@udea.edu.co

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# #### Supongamos que tenemos un dataset de un problema de clasificación

# In[3]:


from library.regularization import plot_ellipse
Cov = np.identity(2) * 1.1
Cov2 = np.array([[1.1,0.5],[0.5,1.1]])
Mean = [1.1,2.1]
Mean2 = [4.1,4.1]
ax = plt.subplot(111)
x, y  = np.random.multivariate_normal(Mean, Cov, 100).T
x2, y2  = np.random.multivariate_normal(Mean2, Cov2, 100).T
ax.plot(x,y,'o',alpha= 0.5)
ax.plot(x2,y2,'o',alpha= 0.5)
ax.axis('equal')

plot_ellipse(ax,Mean,Cov,color='b')
plot_ellipse(ax,Mean2,Cov2)
plt.grid()


# ### ¿Cómo se vería la frontera de clasificación usando un FDG?

# In[4]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
N = 100
x, y  = np.random.multivariate_normal(Mean, Cov, N).T
x2, y2  = np.random.multivariate_normal(Mean2, Cov2, N).T
X = np.r_[np.c_[x,y],np.c_[x2,y2]]
Y = np.r_[np.ones((N,1)),np.zeros((N,1))]
clf = QuadraticDiscriminantAnalysis()
clf.fit(X,Y.flatten())
plt.scatter(X[:,0],X[:,1],c=Y.flatten(), cmap='Set2',alpha=0.5)

h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Blues)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid()


# In[5]:


from library.regularization import Fronteras
clf = QuadraticDiscriminantAnalysis()
Fronteras(clf,100,10)


# y si cambiamos el tipo de modelo...

# In[6]:


from library.regularization import Kernel_classifier
clf = Kernel_classifier(bandwidth=0.2)
Fronteras(clf,100,10)


# **Consultar**: Criterios para seleccionar modelos según su complejidad: 
# - Akaike Information Criterion
# - Bayesian Information Criterion
# - Minimum Description Length
# 
# ¿Qué limitaciones tienen dichos criterios?

# ## Bias vs Variance

# **Bias (sesgo)**: es la diferencia entre la predicción promedio de muestro modelo y el valor correcto que se pretende predecir. Un modelo con alto bias no captura las relaciones relevantes entre las características de entrada y las variables objetivo de salida, pone poca atención a los datos de entrenamiento y sobre-simplifica el modelo.
# 
# **Variance**: es un error debido a una alta sensibilidad a pequeñas fluctuaciones en el conjunto de entrenamiento. Una varianza alta puede causar que el modelo se centre en el ruido contenido en los datos más que en la salida deseada. Los modelos que cometen error por varianza suelten tener buenas desempeños en entrenamiento pero altas tasasa de error en conjuntos de prueba.
# 
# Formalmente:
# 
# El sistema que queremos modelar está dado por:
# 
# $$y=f({\bf{x}}) + e$$
# 
# donde $e$ es el término de error el cual se asume distribuido normalmente con media 0.
# 
# $$Err({\bf{x}}) = E\left[ \left(y - \hat{f}({\bf{x}})\right)^2 \right]$$
# 
# Usando propiedades del valor esperado:
# 
# $$Err({\bf{x}}) = \left( E[\hat{f}({\bf{x}})] - f({\bf{x}})\right)^2 + E\left[\left(\hat{f}({\bf{x}}) - E\left[\hat{f}({\bf{x}})\right]\right)^2\right] + \sigma_e^2$$
# 
# $$Err({\bf{x}}) = \text{Bias}^2 + \text{Variance} + \text{Irreductible Error}$$

# In[7]:


from IPython.display import Image
Image("./Images/biasVVariance.png", width = 600)


# In[8]:


Image("./Images/tradeoff.png", width = 600)


# ### Veamos un ejemplo:

# In[9]:


from library.regularization import PolynomialLinearRegression
def f(size):
    '''
    Returns a sample with 'size' instances without noise.
    '''
    x = np.linspace(0, 4.5, size)
    y = 2 * np.sin(x * 1.5)
    return (x,y)

def sample(size):
    '''
    Returns a sample with 'size' instances.
    '''
    x = np.linspace(0, 4.5, size)
    y = 2 * np.sin(x * 1.5) + np.random.randn(x.size)
    return (x,y)
    
size = 50
f_x,f_y = f(size)
plt.plot(f_x, f_y)
x, y = sample(50)
plt.plot(x, y, 'k.')
model = PolynomialLinearRegression(degree=8)
model.fit(x,y)
p_y = model.predict(x)
plt.plot(f_x, f_y, label="true function")
plt.plot(x, y, 'k.', label="data")
plt.plot(x, p_y, label="polynomial fit")
plt.legend();
plt.grid();


# In[10]:


plt.figure(figsize=(18,3))
for k, degree in enumerate([3, 5, 10, 18]):
    plt.subplot(1,4,k+1)
    n_samples = 20
    n_models = 20
    avg_y = np.zeros(n_samples)
    for i in range(n_models):
        (x,y) = sample(n_samples)
        model = PolynomialLinearRegression(degree=degree)
        model.fit(x,y)
        p_y = model.predict(x)
        avg_y = avg_y + p_y
        plt.plot(x, p_y, 'k-', alpha=.1)
    avg_y = avg_y / n_models
    plt.plot(x, avg_y, 'b--', label="average model")
    plt.plot(x, f(len(x))[1], 'b--', color="red", lw="3", alpha=.5, label="actual function")
    plt.legend();
    plt.grid();
    plt.title("degree %d"%degree)


# In[11]:


from numpy.linalg import norm
n_samples = 20
f_x, f_y = f(n_samples)
n_models = 100
max_degree = 15
var_vals =[]
bias_vals = []
error_vals = []
for degree in range(1, max_degree):
    avg_y = np.zeros(n_samples)
    models = []
    for i in range(n_models):
        (x,y) = sample(n_samples)
        model = PolynomialLinearRegression(degree=degree)
        model.fit(x,y)
        p_y = model.predict(x)
        avg_y = avg_y + p_y
        models.append(p_y)
    avg_y = avg_y / n_models
    bias_2 = norm(avg_y - f_y)/f_y.size
    bias_vals.append(bias_2)
    variance = 0
    for p_y in models:
        variance += norm(avg_y - p_y)
    variance /= f_y.size * n_models
    var_vals.append(variance)
    error_vals.append(variance + bias_2)
plt.plot(range(1, max_degree), bias_vals, label='bias')
plt.plot(range(1, max_degree), var_vals, label='variance')
plt.plot(range(1, max_degree), error_vals, label='error = bias+variance')
plt.legend()
plt.xlabel("polynomial degree")
plt.grid();


# -------------------

# # <font color='blue'>Metodologías de validación</font>
# 
# Cuando vamos a resolver un problema de Machine Learning, tenemos un solo conjunto de datos $\mathcal{D} = \{({\bf{x}}_i,y_i)\}_{i=1}^N$. Las metodologías de validación nos permiten usar ese conjunto de manera apropiada para realizar la selección de los parámetros del modelo y estimar medidas de desempeño confiables.
# 
# 
# Existen varias maneras de muestrear los datos, las dos metodologías más utilizadas son validación cruzada y Bootstrapping.

# ### Validación cruzada ($k$-fold cross-validation)
# 
# En primer lugar se divide el conjunto de datos de manera aleatoria en dos subconjuntos: Training y Test, típicamente 80% - 20% respectivamente. El conjunto de entrenamiento a su vez se divide nuevamente de manera aleatoria en $k$ subconjuntos disyuntos, se usan $k-1$ suubconjuntos para entrenar y el conjunto restante para validar; dicho proceso se repite $k$ veces. El proceso de entrenamiento y validación se utliza para seleccionar los hiperparámetros del modelo y el conjuto de test para evaluar el desempeño una vez escogido el mejor subconjunto de parámetros.

# In[12]:


Image("./Images/grid_search_cross_validation.png", width = 600)


# Imagen tomada de este [sitio](https://scikit-learn.org/stable/modules/cross_validation.html).

# In[13]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def FronterasCV(X,Y, cv):
    nf = cv.get_n_splits(X)
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Acc = np.zeros((4,nf))
    fold = 0
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,5))
    for train_index, val_index in cv.split(X):
        fold += 1
        #------------------------------------
        Xtrain = X[train_index,:]
        Ytrain = Y[train_index]
        Xval = X[val_index,:]
        Yval = Y[val_index]
        #------------------------------------
        for i,n_neighbors in enumerate([1,3,5,7]):
            if fold == 1:
                ax[i].scatter(X[:,0],X[:,1],c=Y.flatten(), cmap='Set2',alpha=0.5)
                ax[i].set_title('Fronteras para k='+str(n_neighbors))
                ax[i].set_xlabel('$x_1$')
                ax[i].set_ylabel('$x_2$')
                ax[i].grid()
            
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(Xtrain,Ytrain.flatten())
            Ypred = clf.predict(Xval)
            Acc[i,fold-1] = accuracy_score(Yval,Ypred)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax[i].contour(xx, yy, Z, cmap=plt.cm.Blues)
        
    plt.show()
    return(Acc)


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
N = 1000
Cov = np.identity(2) * 1.1
Cov2 = np.array([[1.1,0.5],[0.5,1.1]])
Mean = [1.1,2.1]
Mean2 = [4.1,4.1]
x, y  = np.random.multivariate_normal(Mean, Cov, N).T
x2, y2  = np.random.multivariate_normal(Mean2, Cov2, N).T
X = np.r_[np.c_[x,y],np.c_[x2,y2]]
Y = np.r_[np.ones((N,1)),np.zeros((N,1))]
#----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

kf = KFold(n_splits=5)
Acc = FronterasCV(X_train,y_train, kf)
print('Accuracy en cada fold = '+ np.array_str(Acc))
print('Accuracy promedio = '+str(np.mean(Acc,axis=1)) + '+/-' +str(np.std(Acc,axis=1)*2))


# El mejor modelo fue para un $k=7$

# In[15]:


clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train,y_train.flatten())
y_pred = clf.predict(X_test)
print('Accuracy en test = ' + str(accuracy_score(y_test,y_pred)))


# Normalmente el error en test es un poco mayor que en validación, por eso es necesario no quedarse con el error de validación ya que puede mostrar **resultados optimistas**.

# ### Leave-one-out
# 
# Este un caso particular de la validación cruzada en la que se crean tantos folds como muestras hay en el conjunto de datos. Se usa en casos en los que el conjunto de muestras es muy pequeño y se intenta proveer al algoritmo de entrenamiento con el máximo número posible de muestras (todas menos 1) y se valida con la muestra restante. 

# In[ ]:


from sklearn.model_selection import LeaveOneOut

#ó

kf = KFold(n_splits=n)


# ### Validación Bootstrapping (shuffle-split)
# 
# En este la partición de las muestras entre entrenamiento y validación se realiza utilizando aleatoriamente definiendo un porcentaje para entrenamiento/validación y un número de repeticiones. La diferencia fundamental con la metodología anterior es que en el caso de Bootstrapping es posible que una misma muestra se repita en dos subconjuntos de validación. Adicionalmente en el caso de validación cruzada los porcentajes de entrenamiento y validación están definidos implícitamente por el número de folds, mientras que en Bootstrapping no.

# In[17]:


Image("./Images/bootstrap_concept.png", width = 600)


# In[18]:


from sklearn.model_selection import ShuffleSplit
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
Acc = FronterasCV(X_train,y_train, rs)
print('Accuracy en cada fold = '+ np.array_str(Acc))
print('Accuracy promedio = '+str(np.mean(Acc,axis=1)) + '+/-' +str(np.std(Acc,axis=1)*2))


# Aunque en esta prueba se usó un 5% menos de muestras en el entrenamiento de cada fold, los resultados son muy similares al caso anterior, el mejor valor para el hiperparámetro $k$ es 7.

# ### Leave-p-out
# 
# Este un tipo de validación en la que no se define un porcentaje para el conjunto de validación, sino un número $p$ de muestras para validación y las restantes $n-p$ quedan para el entrenamiento. En este caso el número de repeticiones estará definido por el número de combinaciones posibles.

# In[19]:


X=np.random.randn(10,2)


# In[20]:


from sklearn.model_selection import LeavePOut
lpo = LeavePOut(2)
lpo.get_n_splits(X)


# Que corresponde al número de combinaciones posibles N combinado 2.

# In[21]:


from itertools import combinations 
len(list(combinations(range(X.shape[0]), 2)))

LeavePOut(p=1) es igual a LeaveOneOut()
# ## Metodología de validación para problemas desbalanceados
# 
# 
# Si tenemos problemas desbalanceados y usamos una metodología de validación estándar, podemos tener problemas porque la clase minoritaria queda muy mal representada en el conjunto de training.

# In[22]:


N = 1000
x, y  = np.random.multivariate_normal(Mean, Cov, int(N/10)).T
x2, y2  = np.random.multivariate_normal(Mean2, Cov2, N).T
X = np.r_[np.c_[x,y],np.c_[x2,y2]]
Y = np.r_[np.ones((int(N/10),1)),np.zeros((N,1))]


# In[31]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.bar(0,np.sum(Y==0)/Y.shape[0])
plt.bar(1,np.sum(Y==1)/Y.shape[0])
plt.title('Distribución de clases original')
for i in range(2):
    plt.text(i, np.sum(Y==i)/Y.shape[0], str(round(np.sum(Y==i)/Y.shape[0],3)), color='black', fontweight='bold')
#--------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=None)
plt.subplot(1,2,2)
plt.bar(0,np.sum(y_train==0)/y_train.shape[0])
plt.bar(1,np.sum(y_train==1)/y_train.shape[0])
for i in range(2):
    plt.text(i, np.sum(y_train==i)/y_train.shape[0], str(round(np.sum(y_train==i)/y_train.shape[0],3)), color='black', fontweight='bold')
plt.title('Distribución de clases para el entrenamiento')
plt.show()


# Para garantizar que se mantenga la proporción del conjunto de datos original, se debe usar un versión **estratificada** de la metodología de validación:
# 

# In[32]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.bar(0,np.sum(Y==0)/Y.shape[0])
plt.bar(1,np.sum(Y==1)/Y.shape[0])
plt.title('Distribución de clases original')
for i in range(2):
    plt.text(i, np.sum(Y==i)/Y.shape[0], str(round(np.sum(Y==i)/Y.shape[0],3)), color='black', fontweight='bold')
#--------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
plt.subplot(1,2,2)
plt.bar(0,np.sum(y_train==0)/y_train.shape[0])
plt.bar(1,np.sum(y_train==1)/y_train.shape[0])
for i in range(2):
    plt.text(i, np.sum(y_train==i)/y_train.shape[0], str(round(np.sum(y_train==i)/y_train.shape[0],3)), color='black', fontweight='bold')
plt.title('Distribución de clases para el entrenamiento')
plt.show()


# In[ ]:


#Versión de validación cruzada estratificada
from sklearn.model_selection import StratifiedKFold


# In[ ]:


#Versión de validación Bootstrapping estratificada
from sklearn.model_selection import StratifiedShuffleSplit


# ## Metodología de validación por grupos
# 
# 
# Existen problemas de ML en los que todas las muestras de entrenamiento no pueden ser consideradas independientes entre ellas, porque provienen de una fuente común. Existen dos casos:
# 
# - Las muestras de entrenamiento provienen de la misma fuente en tiempos diferentes
# - El objeto sobre el cual queremos hacer predicciones están compuestos por varios vectores de características (**Multi-instance learning**)
# 
# **Ejemplo**: Si se quiere diseñar un sistemas de apoyo diagnóstico para la detección de Parkinson usando grabaciones de voz, es posible que en la base de datos que se usará para el entrenamiento del sistemas, se tengan varias grabaciones del mismo paciente tomadas en diferentes sesiones. Si no se tiene en cuenta ese factor y las muestras se tratan como independientes, puede suceder que muestras de un mismo paciente se encuentren tanto en el conjunto de entrenamiento como en el de validación, por lo que los resultados estarán sesgados de manera optimista.
# 
# Fuente de datos: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Parkinson) 

# In[33]:


import pandas as pd
df = pd.read_csv("DataFiles/parkinsons.data",delimiter=',')
df[:10]


# In[34]:


df.shape


# In[35]:


df[['name','R','Subject','Session']]= df['name'].str.split("_",expand=True)
df = df.drop(['name', 'R', 'Session'], axis=1)


# In[36]:


df[:10]


# In[37]:


len(np.unique(df['Subject']))


# In[38]:


Y = df['status'].values
Pacientes = df['Subject'].values
df = df.drop(['status', 'Subject'], axis=1)
X = df.values


# In[39]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 0).split(df,Y, Pacientes))


# ### Probemos asumiendo independencia

# In[40]:


from sklearn.preprocessing import StandardScaler

#GridSearch

scaler = StandardScaler()
Xtrain = scaler.fit_transform(X[train_inds,:]) #Para usarlo correctamente en un el GridSearch debemos definir un pipeline
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)

#--------------------------------------------------------------------------------------
# Set the parameters by cross-validation
tuned_parameters = [{'n_neighbors': [1,3,5,7,9], 'metric': ['minkowski'], 'p':[1,2]},
                    {'n_neighbors': [1,3,5,7,9], 'metric': ['chebyshev']}]
scores = ['precision', 'recall']
clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=rs, scoring='accuracy')
#--------------------------------------------------------------------------------------
clf.fit(Xtrain, Y[train_inds])


# In[41]:


print(clf.best_params_)
print('Accuracy validación = '+str(clf.best_score_))


# In[42]:


Xtest = scaler.transform(X[test_inds,:])
y_pred = clf.predict(Xtest)
print('Accuracy test = '+str(accuracy_score(Y[test_inds],y_pred)))


# #### Noten la gran diferencia entre el desempeño en validación y el desempeño en test!

# ### Ahora probemos teniendo en cuenta los pacientes

# In[43]:


rs = GroupShuffleSplit(test_size=.25, n_splits=5, random_state = 0).split(Xtrain, Y[train_inds], Pacientes[train_inds])
clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=rs, scoring='accuracy')
clf.fit(Xtrain, Y[train_inds])


# In[44]:


print(clf.best_params_)
print('Accuracy validación = '+str(clf.best_score_))


# In[45]:


Xtest = scaler.transform(X[test_inds,:])
y_pred = clf.predict(Xtest)
print('Accuracy test = '+str(accuracy_score(Y[test_inds],y_pred)))


# **Podemos observar que**:
# 
# - La diferencia entre el error de validación y erro de test es pequeña, lo que muestra consistencia en el resultaDO.
# - El modelo quedo mejor ajustado al problema real y el desempeño en test es 5 punto porcentuales más alto en este caso que en el anterior.

# También existe la variante validación cruzada por grupos: 

# In[ ]:


from sklearn.model_selection import GroupKFold


# Para lo que no existe una implementación curada en sklearn es para la variante **Stratified Group Shuffle Split** o Stratified GroupKFold, pero en el [este](https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation) enlace pueden encontrar una implementación. 

# -----------------

# # <font color='blue'>Curva de aprendizaje</font>

# Definir los porcentajes de muestras para entrenamiento y test no es una tarea sencilla y el valor apropiado dependerá del conjunto de datos con el que estemos trabajando. Una herramienta muy útil para establecer el valor adecuado de muestras de entrenamiento es construir la curva de aprendizaje.
# 
# #### Veamos un ejemplo con los mismos datos del ejemplo anterior.

# In[46]:


from library.learning_curve import plot_learning_curve


# In[47]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
scaler = StandardScaler()
Xtrain = scaler.fit_transform(X)

fig, axes = plt.subplots(3, 2, figsize=(15, 15))

title = "Learning Curves (Random Forest, n_estimators=20)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

#estimator = KNeighborsClassifier(n_neighbors=9, metric='minkowski', p=1)
estimator = RandomForestClassifier(n_estimators=20 ,max_depth=3, random_state=0)
plot_learning_curve(estimator, title, Xtrain, Y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, Y, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()


# Podemos observar que los resultados muestran comportamientos de alta varianza, es decir con cierto nivel de sobre ajuste y una clara necesidad de contar con una base de datos más grande.

# El anterior análisis está hecho asumiendo que las muestras son i.i.d y usando la función:

# In[ ]:


from sklearn.model_selection import learning_curve


# para usar **GroupKFold** o un iterador similar, que es lo recomendable en este caso, debemos realizar manualmente los diferentes experimentos para la construcción de las curvas.

# ### Prueba con otro conjunto de datos

# In[49]:


from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

X, y = load_digits(return_X_y=True)

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()


# En este caso vemos que el modelo SVM alcanza baja varianza y bajo bias, además se requieren más de 1200 muestras para el entrenamiento del modelo.
