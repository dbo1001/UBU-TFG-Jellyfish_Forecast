#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import GridSearchCV
from time import time


# # Pruebas

# In[ ]:


# # pruebas

# from sklearn import datasets
# from sklearn import svm

# X, y = datasets.load_iris(return_X_y=True)
# X.shape, y.shape

# from sklearn.model_selection import cross_val_score
# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, X, y, cv=5)
# scores


# In[ ]:


# # con datos de medusas

# X = pd.read_pickle('../pkls/dfAtributosNormalizado_0_dias_2_celdas.pkl')
# Y = pd.read_pickle('../pkls/dfAvistamientos.pkl')
# Y = np.ravel(Y)

# clf = svm.SVR(kernel='linear', C=1)
# scores = cross_val_score(clf, X, Y, cv=5)
# scores


# In[ ]:


# from sklearn.model_selection import KFold

# kf = KFold(n_splits=2)
# kf.get_n_splits(X)

# print(kf)

# for train_index, test_index in kf.split(X):
# #     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X.loc[train_index], X.loc[test_index]
#     y_train, y_test = Y[train_index], Y[test_index]


# # Algoritmos

# In[15]:


# Carga datos

X = pd.read_pickle('../pkls/dfAtributosNormalizado_0_dias_2_celdas.pkl')
Y = pd.read_pickle('../pkls/dfAvistamientos.pkl')
Y = np.ravel(Y)
X.shape,Y.shape


# In[7]:


# Validacion cruzada

from sklearn.model_selection import KFold

kf = KFold(n_splits=3)
kf.get_n_splits(X)


# ## Random forest

# In[30]:


from sklearn.ensemble import RandomForestRegressor


# In[69]:


def forest(atributos,resultado,k_n):
    params = {'bootstrap' : ['True', 'False'],
            'n_estimators': [50,100,200,500,1000],
            'max_depth': ['None',5,10,50,100],
            'max_features': [2,5,10,20,50,100,'auto','sqrt', 'log2']}
    inicio = time()
    model_random = GridSearchCV(estimator=RandomForestRegressor(), 
                           cv=k_n,
                           param_grid =params,
                           n_jobs = -1)

    model_random.fit(atributos, resultado)
    
    fin = time()
    tiempo = (fin - inicio)/60
    print('Tiempo empleado para Random Forest: {} minutos'.format(tiempo), flush=True)
    print('Best_params: {}\nBest_score: {}'.format(model_random.best_params_,model_random.best_score_), flush=True)
    
    return tiempo,model_random


# In[89]:


# tiempo , modelo = forest(X,Y,3)


# # Nearest Neighbor

# In[68]:


from sklearn.neighbors import KNeighborsRegressor


# In[70]:


def vecino(atributos,resultado,k_n):

    # params = {'n_neighbors' : list(range(1,atributos.shape[1]))}
    params = {'n_neighbors' : random.sample(range(3, atributos.shape[1]), 10) ,
             'weights':['uniform', 'distance']}

    inicio = time()
    model_vecinos = GridSearchCV(estimator=KNeighborsRegressor(), 
                               cv=k_n,
                               param_grid=params,
                                n_jobs = -1)


    model_vecinos.fit(atributos, resultado)
    
    fin = time()
    tiempo = (fin - inicio)/60
    print('Tiempo empleado para Random Forest: {} minutos'.format(tiempo), flush=True)
    print('Best_params: {}\nBest_score: {}'.format(model_vecinos.best_params_,model_vecinos.best_score_), flush=True)
    
    return tiempo,model_vecinos


# In[87]:


# tiempo , modelo = vecino(X,Y,3)


# # SVM

# In[62]:


from sklearn import svm


# In[71]:


def SVR(atributos,resultado,k_n):

    params = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'),
              'C' : [0.5,1.0,10,100],
              'gamma':['scale', 'auto'],
              'epsilon':[0.2]}

    inicio = time()

    model_SVR= GridSearchCV(estimator=svm.SVR(), 
                           cv=k_n,
                           param_grid=params,
                                n_jobs = -1)

    model_SVR.fit(atributos, resultado)
    
    fin = time()
    tiempo = (fin - inicio)/60
    print('Tiempo empleado para Random Forest: {} minutos'.format(tiempo), flush=True)
    print('Best_params: {}\nBest_score: {}'.format(model_SVR.best_params_,model_SVR.best_score_), flush=True)
    
    return tiempo,model_SVR


# In[83]:


# tiempo , modelo = SVR(X,Y,3)


# # Arboles de decision

# In[10]:


from sklearn.tree import DecisionTreeRegressor


# In[72]:


def arbol_decision(atributos,resultado,k_n):
    
    params = {'max_depth': random.sample(range(3, 200), 20),
              'max_features' : ['auto', 'sqrt', 'log2',None]}

    inicio = time()
    
    model_tree= GridSearchCV(estimator=DecisionTreeRegressor(), 
                           cv=k_n,
                           param_grid=params,
                            n_jobs = -1)


    model_tree.fit(atributos, resultado)
    
    fin = time()
    tiempo = (fin - inicio)/60
    print('Tiempo empleado para Random Forest: {} minutos'.format(tiempo), flush=True)
    print('Best_params: {}\nBest_score: {}'.format(model_tree.best_params_,model_tree.best_score_), flush=True)
    
    return tiempo,model_tree


# In[14]:


# tiempo , modelo = arbol_decision(X,Y,3)


# # Boosting

# In[16]:


from sklearn.ensemble import GradientBoostingRegressor


# In[73]:


def boosting(atributos,resultado,k_n):
    
    params = {'n_estimators': random.sample(range(3, 500), 10),
              'max_depth': random.sample(range(3, 500), 10)}

    inicio = time()
    
    model_boos= GridSearchCV(estimator=GradientBoostingRegressor(), 
                           cv=k_n,
                           param_grid=params,
                           n_jobs = -1)


    model_boos.fit(atributos, resultado)
    
    fin = time()
    tiempo = (fin - inicio)/60
    print('Tiempo empleado para Random Forest: {} minutos'.format(tiempo), flush=True)
    print('Best_params: {}\nBest_score: {}'.format(model_boos.best_params_,model_boos.best_score_), flush=True)
    
    return tiempo,model_boos


# In[20]:


# tiempo , modelo = boosting(X,Y,3)


# # Adasboost

# In[21]:


from sklearn.ensemble import AdaBoostRegressor


# In[74]:


def adaboost(atributos,resultado,k_n):
    
    params = {'n_estimators': random.sample(range(3, 500), 10),
              'loss': ['linear', 'square', 'exponential'],
             'random_state': ['None',1,5,10]}

    inicio = time()
    
    model_boos= GridSearchCV(estimator=AdaBoostRegressor(), 
                           cv=k_n,
                           param_grid=params,
                           n_jobs = -1)


    model_boos.fit(atributos, resultado)
    
    fin = time()
    tiempo = (fin - inicio)/60
    print('Tiempo empleado para Random Forest: {} minutos'.format(tiempo), flush=True)
    print('Best_params: {}\nBest_score: {}'.format(model_boos.best_params_,model_boos.best_score_), flush=True)
    
    return tiempo,model_boos


# In[23]:


# tiempo , modelo = adaboost(X,Y,3)


# # GradientBoostingRegressor

# In[24]:


from sklearn.ensemble import GradientBoostingRegressor


# In[75]:


def grad_boosting(atributos,resultado,k_n):
    
    params = {'n_estimators': random.sample(range(3, 500), 10),
              'max_depth': random.sample(range(3, 500), 10), 
              'max_features':['auto', 'sqrt', 'log2',None],
             'random_state': ['None',1,5,10]}

    inicio = time()
    
    model_boos= GridSearchCV(estimator=GradientBoostingRegressor(), 
                           cv=k_n,
                           param_grid=params,
                           n_jobs = -1)


    model_boos.fit(atributos, resultado)
    
    fin = time()
    tiempo = (fin - inicio)/60
    print('Tiempo empleado para Random Forest: {} minutos'.format(tiempo), flush=True)
    print('Best_params: {}\nBest_score: {}'.format(model_boos.best_params_,model_boos.best_score_), flush=True)
    
    return tiempo,model_boos


# In[26]:


# tiempo , modelo = grad_boosting(X,Y,3)


# # MLP (red neuronal)

# In[116]:


from sklearn.neural_network import MLPRegressor


# In[76]:


def MLP(atributos,resultado,k_n):

    params = {'alpha' : [0.00001,0.0001,0.001],
            'max_iter' : [1000,2000,5000],
            'random_state': [0,1,10]}

    inicio = time()

    model_MLP= GridSearchCV(estimator=MLPRegressor(), 
                           cv=k_n,
                           param_grid=params)


    model_MLP.fit(atributos, resultado)
    
    fin = time()
    tiempo = (fin - inicio)/60
    print('Tiempo empleado : {} minutos'.format(tiempo), flush=True)
    print('Best_params: {}\nBest_score: {}'.format(model_MLP.best_params_,model_MLP.best_score_), flush=True)
    
    return tiempo,model_MLP


# In[118]:


# tiempo , modelo = MLP(X,Y,3)


# ## Ejecutar todos los algoritmos con diferentes combinaciones

# In[80]:


# df para guardar resultados
algoritmos  = ['random_forest','nearest_neighbor',
              'SVR','arbol_decision',
               'MLP','Boosting','Adasboost','GradientBoostingRegressor']
df = pd.DataFrame(index=algoritmos)

def reinicia_df():
    df = pd.DataFrame(index=algoritmos)
    df.to_pickle('resultadosKfold.pkl')

def guarda_resultado(alg,dias,celdas,split,resultado,params):
    df = pd.read_pickle('resultadosKfold.pkl')
    print(alg,dias,celdas,resultado,params)
    nombre_col = '{}_dias_{}_celdas_{}_splits'.format(dias,celdas,split)
    if not nombre_col in df.columns:
        df[nombre_col] = np.nan
        df[nombre_col + '_params'] = np.nan
    df.loc[alg,nombre_col] = resultado
    df.loc[alg,nombre_col + '_params'] = params
    df.to_pickle('resultadosKfold.pkl')
    df.to_excel('resultadosKfold.xlsx')

##
reinicia_df()
guarda_resultado('random_forest',1,2,0,3,'hola')
guarda_resultado('nearest_neighbor',1,2,0,4,'adios')
guarda_resultado('SVR',2,2,0,4,'salu2')
df = pd.read_pickle('resultadosKfold.pkl')
df


# In[81]:


import os
import datetime

splits = [5,10,20]

res = ''
reinicia_df()
resultado = Y
atributos = X

listado_archivos = os.listdir('../pkls/')
df = pd.DataFrame(index=algoritmos)
for i in listado_archivos:
    for split in splits:
        if 'dfAtributosNormalizado' in i :
            atributos = pd.read_pickle('../pkls/{}'.format(i))
            dias,celdas = i[23:25],i[31:33]
            n_iter = 100
            print('\n\n' + i + '\n', flush=True)

            # concateno y guardo en log
            res += str(datetime.datetime.now()) + ' --------- ' +  str(i) + '\n'

            model_result = forest(atributos,resultado,n_iter)
            guarda_resultado('random_forest',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> Random Forest ' + str(model_result) + '\n'

            model_result = vecino(atributos,resultado,n_iter)
            guarda_resultado('nearest_neighbor',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> Vecino mas cercano ' + str(model_result) + '\n'

            model_result = SVR(atributos,resultado,n_iter)
            guarda_resultado('SVR',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> SVR ' + str(model_result) + '\n'

            model_result = arbol_decision(atributos,resultado,n_iter)
            guarda_resultado('arbol_decision',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> arbol_decision ' + str(model_result) + '\n'
       
            model_result = boosting(atributos,resultado,n_iter)
            guarda_resultado('Boosting',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> Boosting ' + str(model_result) + '\n'
            
            model_result = adaboost(atributos,resultado,n_iter)
            guarda_resultado('Adaboost',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> Adaboost ' + str(model_result) + '\n'
            
            model_result = grad_boosting(atributos,resultado,n_iter)
            guarda_resultado('GradientBoostingRegressor',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> GradientBoostingRegressor ' + str(model_result) + '\n'
            
            model_result = MLP(atributos,resultado,n_iter)
            guarda_resultado('MLP',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> MLP ' + str(model_result) + '\n'


        f = open ('log.txt','wb')

        f.write(bytes(res, encoding='utf-8'))
        f.close()


# # Voting regressor

# In[32]:


from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression


# In[56]:


# def voting(atributos,resultado,k_n):

#     inicio = time()
    
#     params = {}
    
#     reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
#     reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
#     reg3 = LinearRegression()
#     ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
    
#     grid = GridSearchCV(estimator=ereg, param_grid=params, cv=k_n)

#     ereg = grid.fit(atributos, resultado)
    
#     fin = time()
#     tiempo = (fin - inicio)/60
#     print('Tiempo empleado : {} minutos'.format(tiempo), flush=True)
    
#     return tiempo,ereg


# In[57]:


# tiempo , modelo = voting(X,Y,3)


# In[58]:


# modelo.best_score_


# In[ ]:




