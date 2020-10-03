#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits import mplot3d

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#df = pd.read_csv("../DataFiles/housing.data",delim_whitespace=True, header=None, names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
USAhousing = pd.read_csv("../DataFiles/USA_Housing.csv")
samples, columns = USAhousing.shape
features = columns -1
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
print("Samples: ",samples, "   Features + y: ",columns)
USAhousing.head(7)


# In[4]:


USAhousing.describe()


# In[5]:


def get_bin(rangex,valuex):
    for i, rang in enumerate(rangex):
        if (valuex) <= (rang):
            return i
    


# In[6]:


def get_y_mean(y, rangey):
    for i in rangey:
       
        if y < i:
            return i


# In[7]:


def get_y(array,rangey):
    total = np.sum(array)
    #print("Total ",total)
    prob = np.zeros((len(array)))
    if total != 0:
        prob = (array/total) 
        #print("Array prob", prob)
        prob = prob*rangey
        #print("value *rangey ",value)
    y_mean = np.sum(prob)
    #print("Prob total: ", prob )
    #y_mean = get_y_mean(prob, rangey)
    
    return(y_mean)


# In[8]:


array = np.arange(1,6,1)
print("original",array)
rangey = np.arange(0,300,60)
print("rangey ",rangey)
t = array/0.5

c = get_y(array, rangey)


# In[9]:


x = t*rangey

sumx = np.sum(x)
print("divide total",t)
print("result *rangey",x)
print("rangey ",rangey)
print("sumtotal ",sumx)


# In[10]:


def create_samples(x,newpoints=1000):
    #newx = np.random.uniform(np.min(x),np.max(x),newpoints)
    newx = np.linspace(np.min(x), np.max(x), num=newpoints)
    newx = np.around(newx,4)
    #newx = np.concatenate((x,newx))
    return newx


# In[11]:


def create_model_hist(x1,x2, y,bins = 10):
    minx1,max1 = np.min(x1), np.max(x1)
    minx2,max2 = np.min(x2), np.max(x2)
    miny,maxy = np.min(y), np.max(y)
    stepx1, stepx2, stepy = np.ptp(x1)/bins, np.ptp(x2)/bins, np.ptp(y)/bins
    
    rangex1, rangex2, rangey = np.arange(minx1+stepx1, max1+stepx1, stepx1), np.arange(minx2+stepx2, max2+stepx2, stepx2), np.arange(miny+stepy,maxy+stepx2,stepy)
    rangex1, rangex2, rangey = np.around(rangex1,4),np.around(rangex2,4), np.around(rangey,4)
    #print ("rango y",rangey)
    #print ("rango x2",rangex2)
    out = np.zeros((bins,bins,bins))
      
    for i,value in enumerate(zip(x1,x2)):
        
        binx = get_bin(rangex1,value[0])
        
        binx2 = get_bin(rangex2,value[1])
        outy = y[i]
        biny = get_bin(rangey,outy)
        
        
        out[binx,binx2,biny]+=1
        
    rangey = np.arange((miny+stepy+miny)/2,(maxy+1),stepy)
    print("Range y: ",rangey.shape)
    model = np.zeros((bins,bins))
    
    for i in range(0,bins):
        for j in range(0,bins):
            col = out[i,j,:] 
            #print("col ",col)
           
            y_mean = get_y(col,rangey)
            
            
            model[i][j] = y_mean
            
   
    print("unique model: ", np.unique(model).shape)
    
    return list((model, rangex1,rangex2,rangey))


# In[27]:


def build_model(model, x1,x2):    
    newy = np.zeros((x1.shape[0],x2.shape[0]))
    y = model[0]
    
    rangex1 = model[1]
    rangex2 = model[2]
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            binx = get_bin(rangex1,x1[i])
            print("x1",binx)
            binx2 = get_bin(rangex2, x2[i])
            print("x2",binx2)
            newy[i,j] = y[binx][binx2]
                        
    return newy


# In[28]:


def plot_hpi(x="Avg. Area House Age", x2="Area Population", num_points = 100):
    zdata = y[0:num_points]
    xdata = X[x].iloc[0:num_points,]
    ydata = X[x2].iloc[0:num_points,]
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    ax.set_xlabel(x)
    ax.set_ylabel(x2,labelpad=0.9)
    ax.set_zlabel("HPI")
    return ax


# In[29]:



def plot_hpi_regression(X,Y,Z):
    num_points = 100
    zdata = Z#y[0:num_points]
    xdata = X#X[0:num_points,]
    ydata = Y#Y[0:num_points,]

    

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.summer,
                               linewidth=0.2, antialiased=False)
    
    #ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
   
    plt.show()


# In[30]:


def plot_hpi_regression2(X,Y,Z):
    

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    
    
    X, Y = np.meshgrid(X, Y)
    
    # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z,
    #                            linewidth=0.2, antialiased=True)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               linewidth=0.3, antialiased=False)
    
   
    plt.show()


# In[31]:


plot_h = plot_hpi(num_points=10)


# In[32]:


np.diff(model[3])


# In[41]:


x1 = X['Avg. Area House Age']
x1 = np.array(x1)

x2 = X['Area Population']
ycopy = y.copy()

model = create_model_hist(x1,x2,ycopy)


# In[42]:


newx1 = create_samples(x1,newpoints=100)
newx2 = create_samples(x2,newpoints=100)
print(newx1.shape,newx1.shape)


# In[43]:


newy = build_model(model,newx1,newx2)
print("shape newy", newy.shape, " unique ",np.unique(newy).shape)


# In[37]:


model[2]


# In[38]:


model[0]


# In[39]:


newx1


# In[35]:


get_bin(model[1],newx1[99])


# In[40]:


plot_hpi_regression(newx1,newx2,newy)


# In[ ]:




