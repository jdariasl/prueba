#!/usr/bin/env python
# coding: utf-8

# # Selección de Características

# ### Julián D. Arias Londoño
# 
# Profesor Asociado  
# Departamento de Ingeniería de Sistemas  
# Universidad de Antioquia, Medellín, Colombia  
# julian.ariasl@udea.edu.co

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Introducción

# En anteriores sesiones hemos discutido algunas razones por las cuales puede ser necesario reducir el número de variables en un problema de aprendizaje automático. Principalmente hemos aludido a la necesidad de llevar a cabo un proceso de reducción de dimensión, debido al problema conocido como "maldición de la dimensionalidad", sin embargo, son múltiples los beneficios que podemos obtener.
# 
# En primer lugar, si estamos usando un modelo paramétrico, usualmente el número de parámetros que deben ser ajustados durante el entrenamiento es proporcional al número de variables, razón por la cual si reducimos la dimensión del espacio de características, estaremos a su vez reduciendo la complejidad del modelo.
# 
# Algunos beneficios adicionales pueden ser:
# 
# <ul>
# <li>Simplificar el análisis de resultados</li>
# <li>Mejorar el desempeño del sistema a través de una representación más estable</li>
# <li>Remover información redundante o irrelevante para el problema</li>
# <li>Descubrir estructuras subyacentes en los datos, o formas de representación gráfica más simples</li>
# </ul>

# Como ya fue comentando en sesiones anteriores, existen principalmente dos estrategias para reducir el número de variables: <b>selección de características</b> y <b>extracción de características</b>. Ambas requieren la definición de un criterio, en el caso de selección el criterio está asociado a encontrar el mejor subconjunto de variables, de todos los posibles subconjuntos. Mientras que en extracción el criterio está asociado a encontrar la mejor transformación (combinación de variables) sobre todas las transformaciones posibles. Por ahora nos centraremos en la estrategia de selección y en sesiones posteriores revisaremos las estrategias básicas de extracción.

# ### Ventajas de la selección de variables

# <ul>
# <li>Reducir variables que pueden ser costosas de obtener en términos computacionales</li>
# <li>Extraer reglas de clasificación o regresión, que conserven el sentido "físico" a partir del modelo, teniendo en cuenta que las características conservan su interpretación original.</li>
# <li>Manejo de características no numéricas</li>
# </ul>

# ---------------------------------------------------------------------------------------------

# En primer lugar es necesario clarificar porqué razón es necesario llevar a cabo un análisis en conjunto de todas las variables, en lugar de realizar análisis individuales. La gráfica siguiente representa la capacidad discriminante de cuatro variables en un problema de clasificación.

# ![image alt >](./Images/Var12.png)
# ![image alt <](./Images/Var34.png)

# Si realizamos un análisis individual, por ejemplo basado en el índice de correlación o el índice discriminante de Fisher, el resultado indicará que la mejor variable es la 1 y que la peor es la 4. Sin embargo, si el análisis evalúa diferentes subconjuntos de variables, podría darse cuenta que la únión de esas dos variables obteniene un resultado que, en conjunto, es mejor que cualquiera de las variables individuales, por lo que no sería una buena decisión eliminar la variable 4 usando como criterio únicamente un análisis individual de la capacidad discriminante de dicha característica.

# ### Problema

# Dado un conjunto de variables $d$, ¿cuál es el mejor subconjunto de variables de tamaño $p$? 

# Evaluar el criterio de optimalidad para todas las posibles combinaciones de $p$ variables seleccionadas a partir de $d$ variables, implica evaluar un número de combinación igual a:

# $$n_p = \frac{d!}{(d-p)!p!}$$

# el cual puede ser muy elevado incluso para valores moderados de $d$ y $p$, por ejemplo, seleccionar las mejores 10 características de un conjunto total de 25, implica evaluar 3.268.760 subconjuntos diferentes de características, para lo cual se debió también evaluar el criterio de optimalidad en cada uno de ellos. Adicionalmente no existe un criterio para seleccionar $p$, razón por la cual el número de posibles combinaciones que deberían ser evaluadas puede crecer exponencialmente.

# A el análisis descrito en el párrafo anterior se le conoce como "<b>Fuerza bruta</b>", y aunque entregaría el mejor resultado, no puede ser llevado a cabo en tiempos razonables, razón por la cual fue necesario desarrollar <b>métodos de búsqueda</b> cuyo objetivo es encontrar el mejor subconjunto de variables (aunque no pueden garantizar que lo encontrarán), sin necesidad de evaluar todas las posibles combinaciones de características.

# Los métodos subóptimos de selección de variables, constan de dos componentes: un criterio de selección y una estrategia de búsqueda.

# ## Criterios de selección

# <ul>
# <li><b>Filtro</b>: La función objetivo evalúa el subconjunto de características a partir de su contenido de información, típicamente se utiliza alguna distancia entre clases, medidas de dependiencia estadística o medidas basadas en teoría de la información.</li>
#     <li><b>Wrapper</b>: La función objetivo es un modelo de aprendizaje, el cual evalúa el subconjunto de características a partir de su capacidad predictiva ($1-Error$ en los datos de prueba), usando una metodología de validación apropiada.</li>
# </ul>

# ### Criterios Tipo Filtro

# A continuación veremos algunos ejemplos de funciones criterio tipo filtro que pueden usarse:

# #### Distancia probabilística

# La distancia probabilística mide la distancia entre dos distribuciones $p({\bf{x}}|c_1)$ and $p({\bf{x}}|c_2)$ y puede ser usada para la selección de características en problemas de clasificación: 

# $$J_D(c_1,c_2) = \int [p({\bf{x}}|c_1) - p({\bf{x}}|c_2)]\log \left( \frac{p({\bf{x}}|c_1)}{p({\bf{x}}|c_2)} \right) $$

# Si se usa una distribución normal para describir las clases, como por ejemplo en las funciones discriminantes Gaussianas, la integral da como resultado:

# $$J_D = \frac{1}{2}(\mu_1 - {\bf{\mu}}_2)^T (\Sigma_1^{-1} + \Sigma_2^{-1})(\mu_1 - \mu_2) + Tr \{ 2\Sigma_1^{-1}\Sigma_2 - 2I \}$$

# Si el problema es de múltiples clases, existen variantes que pueden utilizarce, por ejemplo: 

# $$J = \max_{i,j} J_D(c_i,c_j )$$

# $$J = \sum_{i < j} J_D(c_1,c_2)p(c_1)p(c_2)$$

# #### Distancia entre clases

# Se pueden usar diferentes medidas de distancia, por ejemplo distancia Euclidiana, Mahalanobis (Consultar), o por ejemplo una medida basada en el índice de Fisher, que utiliza el concepto de dispersión entre clases ($S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T$) y de dispersión intra clase ($S_W = (\Sigma_1 + \Sigma_2)$), para definir el criterio:

# $$J = \frac{Tr\{S_B\}}{Tr\{S_W\}}$$

# #### Criterios basados en correlación y en medidas de teoría de la información

# Este tipo de criterios están basados en la suposición de que los subconjuntos de características óptimos, contienen características altamente correlacionadas con la variable de salida y no correlacionadas con las demás variables de entrada. El mismo concepto visto en clases anteriores. Un posible criterio sería:

# $$ J = \frac{\sum_{i=1}^{p}\rho_{ic}}{\sum_{i=1}^{p}\sum_{j=i+1}^{p}\rho_{ij}}$$

# donde $\rho$ es el coeficiente de correlación entre las variables indicadas por los subíndices, siendo $c$ la variable de salida (variable a predecir). El coeficiente de correlación tiene la habilidad de medir el nivel de relación entre dos variables, pero únicamente evalúa la relación lineal. Una medida más robusta debería incluir relaciones no lineales, por ejemplo la <b> Información Mutua</b> es una medida de relación no lineal definida como:

# $$J = I(X_m;c) = H(c) - H(c|X_m)$$

# donde $I(X_m;c)$ es la información mutua entre el subconjunto de variables $X_m$ y la variable de salida $c$, $H(c)$ es la entropía de la variable de salida $c$ y $H(c|X_m)$ es la entropía condicional de $c$, dado que se conoce $X_m$. En palabras, la información mutua corresponde a la reducción en la incertidumbre de la variapre $c$ debido al conocimiento de las variables incluidas en el subconjunto $X_m$. Como vimos en las clases sobre árboles de deicisón, la entropía es en realidad un funcional, es una función que tiene como entrada otra función, la cual corresponde a la distribución de probabilidad de la variable bajo análisis. Por lo tanto la implementación de la Información mutua, depende del tipo de función de distribución que se asuma para cada una de las variables. 

# ### Ventajas y Desventajas de cada uno de los tipos de criterio 

# ### Filtro

# #### Ventajas

# <ul>
# <li>Rápida ejecución. Los filtro involucran generalmente cálculos no iterativos relacionados con el conjunto de datos, por lo cual son mucho más rápidos que el entrenamiento de un modelo de predicción.</li>
# <li>Generalidad. Debido a que los filtros evalúan las propiedades intrínsecas de los datos, más que las interacciones con un modelo de aprendizaje particular, sus resultados exhiben más generalidad, es decir que la solución puede ser "buena" para una familia más grande de modelos.</li>
# </ul>

# #### Desventajas

# <ul>
# <li>Tendencia a seleccionar subconjuntos de características grandes. Debido a que las funciones objetivo son usualmente monótonas, el filtro tiende a seleccionar el conjunto completo de variables como el mejor.</li>
# </ul>

# ### Wrapper

# #### Ventajas

# <ul>
# <li>Exactitud. Los wrappers generalmente alcanzan mejores tasas de predicción que los filtros, debido a que ellos están ajustados especifícamente para reducir el error de validación.</li>
# <li>Capacidad de generalización. Debido a que los criterios wrappers usan una metodología de validación, tienen la capacidad de evitar el sobre ajuste y proporcionar mejor capacidad de generalización.</li>
# </ul>

# #### Desventajas

# <ul>
# <li>Ejecución lenta. Debido a que el wrapper debe entrenar un clasificador por cada subconjunto de variables (o varios si se usa validación cruzada), el costo computacional puede ser muy alto.</li>
# <li>Falta de generalidad. Debido a que el criterio wrapper usa un modelo de predicción específico, el subconjunto de variables finalmente seleccionado, puede ser bueno para el modelo específico usado como criterio, pero no tan bueno para otros modelos.</li>
# </ul>

# ----------------------------------------------------------------------------------------------------------------

# ## Estrategias de búsqueda

# ### 1. Selección secuencial hacia adelante (Sequential Forward Selection - SFS)

# En este método se comienza con un subconjunto de características vacío y se van adicionando características, una a la vez, hasta que se alcanza el conjunto final con el mayor criterio $J$. La característica adicionada en cada paso, es aquella que con la cual se maximice el criterio de selección.

# #### Algoritmo

#  <ol>
#   <li>Inicializar el conjunto vacío $X_0 = \{\emptyset\}$</li>
#   <li>Seleccionar la siguiente característica $x^+ = \arg\max_{x \notin X_k } \left[ J(X_k + x)\right] $</li>
#   <li>Actualizar el conjunto de variables $X_{k + 1} = X_k + x^+; \; k=k+1$</li>
#   <li>Volver al paso 2. </li>
# </ol> 

# SFS presenta mejor desempeño cuando el conjunto óptimo tiene un número de características bajo. Sin embargo, su principal desventaja es que el método es incapaz de remover variables que se vuelven obsoletas después de la adición de otras características.

# <b> Ejemplo:</b> Considere la siguiente función como un criterio válido:

# $$J(X) = -2x_1x_2 + 3x_1 + 5x_2 - 2x_1x_2x_3 + 7x_3 + 4x_4 - 2x_1x_2x_3x_4$$

# <b>Solución:</b>

# <img src="./Images/SFS.png" alt="SFS" width="600"/>

# ### 2. Selección secuencial hacia atrás (Sequential Backward Selection - SBS)

# El método SBS es análogo al método anterior pero comenzando con el conjunto completo y eliminando una característica a la vez. La característica eliminada es aquella para la cual el criterio $J$ decresca en menor valor (o incluso aumente).

# #### Algoritmo

#  <ol>
#   <li>Inicializar el conjunto lleno $X_0 = X$</li>
#   <li>Seleccionar la siguiente característica $x^- = \arg\max_{x \in X_k } \left[ J(X_k - x)\right] $</li>
#   <li>Actualizar el conjunto de variables $X_{k + 1} = X_k - x^-; \; k=k+1$</li>
#   <li>Volver al paso 2. </li>
# </ol> 

# SBS presenta mejor desempeño cuando el conjunto óptimo tiene un número de características elevado. Sin embargo, su principal desventaja es que el método es incapaz de reevaluar la utilidad de variables que fueron removidas en iteraciones previas.

# ### 3. Selección Más-L Menos-R (LRS) 

# Este es un método que permite algún nivel de retractación en el proceso de selección de características. Si $L > R$, el algoritmo corresponde a un procedimiento hacia adelante, primero se adicionan $L$ características al conjunto actual usando la estrategia SFS, y posteriormente se remueven las peores $R$ características usando SBS.

# Si por el contrario $L < R$ el proceso es hacia atrá, comenzando con el conjunto completo, removiendo $R$ y posteriormente adicionando $L$ variables.

# #### Algoritmo

#  <ol><li>Evalúe:</li>
#   <ul>
# <li>Si $L > R$ entonces</li>
# <ul>
# <li>Comience con el conjunto vacío $X_0 = \{\emptyset\}$</li>
# </ul>
# <li>De lo contrario:</li>
#   <ul>
# <li>Comience con el conjunto completo $X_0 = X$</li>
# <li>Vaya al paso 3</li>
# </ul>
# </ul>
#   <li>Repita $L$ veces</li>
#   <ul>
# <li>$x^+ = \arg\max_{x \notin X_k } \left[ J(X_k + x)\right] $</li>
# <li>$X_{k + 1} = X_k + x^+; \; k=k+1$</li>
# </ul>
#   <li>Reputa $R$ veces</li>
#    <ul>
# <li>$x^- = \arg\max_{x \in X_k } \left[ J(X_k - x)\right] $</li>
# <li>$X_{k + 1} = X_k - x^-; \; k=k+1$</li>
# </ul>
#   <li>Volver al paso 2. </li>
# </ol> 

# LRS intenta compensar la debilidad de los métodos SFS y SBS con capacidades de retractación. Sin embargo, su principal problema es la introducción de dos parámetros adicionales, $L$ y $R$, y ninguna aproximación teórica que permita ajustarlos.

# ### Busqueda Bidireccional (BDS)

# En este caso los métodos SFS y SBS se ejecutan de manera simultánea, sin embargo para garantizar que el método converge a una solución, se establecen las siguientes reglas:
# <ul>
# <li>Características seleccionadas por SFS para ser añadidas, no pueden ser removidas por SBS</li>
# <li>Características eliminadas por SBS no pueden ser añadidas por SFS</li>
# <li>Si por ejemplo, antes de que SFS intente adicionar una nueva característica, el método evalúa si dicha característica ya fue removida por SBS y, si fue removida previamente, intenta adicionar la segunda mejor variable. SBS opera de manera similar.</li>
# </ul>

# #### Algoritmo

#  <ol><li>Comience SFS con el conjunto vacío $X_F = \{\emptyset\}$</li>
#   <li>Comience SBS con el conjunto completo $X_B = X$</li>
#    <li>Seleccione la mejor característica</li>
#   <ul>
# <li>$x^+ = \arg\max_{x \notin X_{F_k}, x \in X_{B_k} } \left[ J(X_{F_k} + x)\right] $</li>
# <li>$X_{F_{k + 1}} = X_{F_{k}} + x^+; \; k=k+1$</li>
# </ul>
#   <li>Remueva la peor característica</li>
#    <ul>
# <li>$x^- = \arg\max_{x \in X_{B_k}, x \notin X_{F_{k + 1}}} \left[ J(X_B - x)\right] $</li>
# <li>$X_{B_{k + 1}} = X_{B_k} - x^-; \; k=k+1$</li>
# </ul>
#   <li>Volver al paso 2. </li>
# </ol> 

# ### Selección secuencia flotante (Sequential Floating Selection (SFFS y SFBS))

# Este método es una extensión del método LRS, que incorpora propiedades flexibles de retractación. En lugar de fijar los valores de $L$ y $R$ previamente, esté método permite que los valores sean determinados a partir de los datos. La dimensionalidad del conjunto de características seleccionado "flota" hacia arriba y hacia abajo, durante las iteraciones del algoritmo.
# 
# Existen dos métodos flotantes:
# 
# <ul>
# <li> <b>Sequential Floating Forward Selection</b>, el cual comienza con el conjunto vacío, el cual una vez terminado el paso hacia adelante, realiza pasos hacia atrás siempre y cuando se incremente el criterio de selección definido.</li>
# <li> <b>Sequential Floating Backward Selection</b>, el cual comienza con el conjunto completo y en el primer paso elimina variables. De manera análoga a SFFS, esté método adiciona variables en la medida en que éstas incrementen el criterio de selección.</li>
# </ul>

# ### Algoritmo SFFS (SFBS es análogo)

#  <ol>
#      <li>Comience con el conjunto vacío $X_0 = \{\emptyset\}$</li>
#      <li>Seleccione la mejor característica</li>
#          <ul>
#            <li>$x^+ = \arg\max_{x \notin X_{k}} \left[ J(X_{k} + x)\right] $</li>
#            <li>$X_{k + 1} = X_{k} + x^+; \; k=k+1$</li>
#         </ul>
#      <li>Seleccione la peor característica</li>
#         <ul>
#          <li>$x^- = \arg\max_{x \in X_{k}} \left[ J(X_k - x)\right] $</li>
#          </ul>
#      <li> Evalúe: </li>
#          <ul>
#             <li>Si $J(X_k - x^-) > J(X_k)$</li>
#                <ul>
#                   <li>$X_{k + 1} = X_{k} - x^-; \; k=k+1$</li>
#                   <li> Volver al paso 3. </li>
#                </ul>
#             <li>De lo contrario</li>
#               <ul>
#                  <li> Volver al paso 2. </li>
#               </ul>
#           </ul>    
# </ol> 

# In[ ]:




