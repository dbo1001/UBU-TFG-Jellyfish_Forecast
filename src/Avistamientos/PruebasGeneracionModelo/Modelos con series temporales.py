#!/usr/bin/env python
# coding: utf-8

# In[103]:


## imports
import pandas as pd
import numpy as np
import xarray as xr
from time import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# In[104]:


## Para guargar el modelo
'''
from joblib import dump, load
dump(clf, 'filename.joblib') # guardar
clf = load('filename.joblib') # cargar
'''


# In[105]:


## carga datos pruebas

df_atributos = pd.read_pickle('../pkls/dfAtributosNormalizado_7_dias_2_celdas.pkl')
df_avistamientos = pd.read_pickle('../pkls/dfAvistamientos.pkl')
df_avistamientos.head()
df_atributos.head()


# In[106]:


atributos = df_atributos
resultado = np.ravel(df_avistamientos)


# In[107]:


atributos.shape


# In[108]:


## Series temporales
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=10)
# print(tscv)

for train_index, test_index in tscv.split(df_atributos):
#     print("TRAIN:", train_index, "\n\tTEST:", test_index)
    X_train, X_test = df_atributos.iloc[train_index], df_atributos.iloc[test_index]
    y_train, y_test = df_avistamientos.iloc[train_index], df_avistamientos.iloc[test_index]


# In[113]:


# tscv = TimeSeriesSplit(n_splits=10)
# forest(atributos,resultado,2,tscv)
# tscv = TimeSeriesSplit(n_splits=5)
# forest(atributos,resultado,2,tscv)
# tscv = TimeSeriesSplit(n_splits=2)
# forest(atributos,resultado,2,tscv)


# # Random Forest

# In[134]:


## random forest con series temporales

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
def forest(atributos,resultado,n,tscv):
    params = {'n_estimators': randint(1, 200),
               'max_depth': randint(1, 100),
              'max_features': randint(1,atributos.shape[1])}
    inicio = time()
    model_random = RandomizedSearchCV(estimator=RandomForestRegressor(), 
                           cv=tscv,
                           n_iter=n,
                           param_distributions=params,
                                      n_jobs = 4)

    model_random.fit(atributos, resultado)
    fin = time()
    print('{} minutos'.format((fin - inicio)/60), flush=True)
    print('{}\n{}'.format(model_random.best_params_,model_random.best_score_), flush=True)
    return ['{} minutos \n'.format((fin - inicio)/60),model_random.best_params_,model_random.best_score_]


# In[ ]:





# # Nearest Neighbor

# In[135]:


## vecino mas cercano con series temporales

from sklearn.neighbors import KNeighborsRegressor
def vecino(atributos,resultado,n,tscv):

    # params = {'n_neighbors' : list(range(1,atributos.shape[1]))}
    params = {'n_neighbors' : list(range(1,60))}

    inicio = time()
    model_vecinos = RandomizedSearchCV(estimator=KNeighborsRegressor(), 
                           cv=tscv,
                           n_iter=n,
                           param_distributions=params,
                                      n_jobs = 4)


    model_vecinos.fit(atributos, resultado)
    fin = time()
    print('{} minutos'.format((fin - inicio)/60))
    print('{}\n{}'.format(model_vecinos.best_params_,model_vecinos.best_score_))
    return ['{} minutos \n'.format((fin - inicio)/60),model_vecinos.best_params_,model_vecinos.best_score_]


# In[136]:



def vecino_grid(atributos,resultado,n,tscv):

    # params = {'n_neighbors' : list(range(1,atributos.shape[1]))}
    params = {'n_neighbors' : list(range(1,60))}

    inicio = time()
    
    model_vecinos_grid= GridSearchCV(estimator=KNeighborsRegressor(), 
                           cv=tscv,
                           param_grid=params,
                                    n_jobs = 4)


    model_vecinos_grid.fit(atributos, resultado)
    fin = time()
    print('{} minutos'.format((fin - inicio)/60))
    print('{}\n{}'.format(model_vecinos_grid.best_params_,model_vecinos_grid.best_score_))
    return ['{} minutos \n'.format((fin - inicio)/60),model_vecinos_grid.best_params_,model_vecinos_grid.best_score_]


# In[76]:


# model_vecinos.cv_results_


# # SVM

# In[137]:


from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def SVR(atributos,resultado,n,tscv):

    params = {'svr__kernel':('linear', 'rbf'),'svr__C' : [0.5,1.0,10,100], 'svr__epsilon':[0.2]}

    inicio = time()
    steps =  [('scaler',StandardScaler()), ('svr',svm.SVR())]
    pipeline = Pipeline(steps)


    model_SVR= RandomizedSearchCV(estimator=pipeline, 
                           cv=tscv,
                           n_iter=n,
                           param_distributions=params,
                                 n_jobs = 4)


    model_SVR.fit(atributos, resultado)
    fin = time()
    print('{} minutos'.format((fin - inicio)/60))
    print('{}\n{}'.format(model_SVR.best_params_,model_SVR.best_score_))
    return ['{} minutos \n'.format((fin - inicio)/60),model_SVR.best_params_,model_SVR.best_score_]


# In[138]:


# model_SVR.cv_results_


# In[139]:


from sklearn import svm
def SVR_grid(atributos,resultadon,n,tscv):
    params = {'svr__kernel':('linear', 'rbf'),'svr__C' : [0.5,1.0,10,100], 'svr__epsilon':[0.2]}

    inicio = time()
    steps =  [('scaler',StandardScaler()), ('svr',svm.SVR())]
    pipeline = Pipeline(steps)

    model_SVR_grid= GridSearchCV(estimator=pipeline, 
                           cv=tscv,
                           param_grid=params,
                            n_jobs = 4)


    model_SVR_grid.fit(atributos, resultado)
    fin = time()
    print('{} minutos'.format((fin - inicio)/60))
    print('{}\n{}'.format(model_SVR_grid.best_params_,model_SVR_grid.best_score_))
    return ['{} minutos \n'.format((fin - inicio)/60),model_SVR_grid.best_params_,model_SVR_grid.best_score_]


# In[80]:


# model_SVR_grid.cv_results_


# # Árboles de decisión

# In[140]:


from sklearn.tree import DecisionTreeRegressor
def arbol_decision(atributos,resultadon,n,tscv):
    params = {'max_depth':list(range(1,50)),'max_features' : ['auto', 'sqrt', 'log2',None]}

    inicio = time()
    model_tree= RandomizedSearchCV(estimator=DecisionTreeRegressor(), 
                           cv=tscv,
                           n_iter=n,
                           param_distributions=params,
                            n_jobs = 4)


    model_tree.fit(atributos, resultado)
    fin = time()
    print('{} minutos'.format((fin - inicio)/60))
    print('{}\n{}'.format(model_tree.best_params_,model_tree.best_score_))
    return ['{} minutos \n'.format((fin - inicio)/60),model_tree.best_params_,model_tree.best_score_]


# In[141]:


def arbol_decision_grid(atributos,resultadon,n,tscv):
    params = {'max_depth':list(range(1,50)),'max_features' : ['auto', 'sqrt', 'log2',None]}

    inicio = time()
    model_tree= GridSearchCV(estimator=DecisionTreeRegressor(), 
                           cv=tscv,
                           param_grid=params,
                            n_jobs = 4)


    model_tree.fit(atributos, resultado)
    fin = time()
    print('{} minutos'.format((fin - inicio)/60))
    print('{}\n{}'.format(model_tree.best_params_,model_tree.best_score_))
    return ['{} minutos \n'.format((fin - inicio)/60),model_tree.best_params_,model_tree.best_score_]


# In[83]:


# model_tree.cv_results_


# # Red Neuronal

# ## MLP

# In[145]:


# from sklearn.neural_network import MLPRegressor

# params = {'mlp__alpha':('linear', 'rbf'),
#           'mlp__max_iter' : [200,500,2000],
#           'mlp__random_state': [0,1,10]}

# inicio = time()

# steps =  [('scaler',StandardScaler()), ('mlp',MLPRegressor())]
# pipeline = Pipeline(steps)

# model_MLP= GridSearchCV(estimator=pipeline, 
#                        cv=tscv,
#                        param_grid=params)


# model_MLP.fit(atributos, resultado)
# fin = time()
# print('{} minutos'.format((fin - inicio)/60))


# In[36]:


atributos.describe()


# # Boosting

# In[132]:


from sklearn.ensemble import GradientBoostingRegressor
def boosting(atributos,resultadon,n,tscv):
    params = {'n_estimators': randint(1, 200),
          'max_depth': randint(1, 100)}

    inicio = time()
    model_boos= RandomizedSearchCV(estimator=GradientBoostingRegressor(), 
                           cv=tscv,
                           n_iter=n,
                           param_distributions=params,
                           n_jobs = 4)


    model_boos.fit(atributos, resultado)
    fin = time()
    print('{} minutos'.format((fin - inicio)/60))
    print('{}\n{}'.format(model_boos.best_params_,model_boos.best_score_))
    return ['{} minutos \n'.format((fin - inicio)/60),model_boos.best_params_,model_boos.best_score_]


# In[142]:


# boosting(atributos,resultado,2,tscv) 


# # ensembles

# In[ ]:





# In[115]:


# df para guardar resultados
algoritmos  = ['random_forest','nearest_neighbor','nearest_neighbor_grid',
               'SVR','SVR_grid','arbol_decision', 'arbol_decision_grid',
               'MLP','Boosting','ensembles']
df = pd.DataFrame(index=algoritmos)

def reinicia_df():
    df = pd.DataFrame(index=algoritmos)
    df.to_pickle('resultados.pkl')

def guarda_resultado(alg,dias,celdas,split,resultado,params):
    df = pd.read_pickle('resultados.pkl')
    print(alg,dias,celdas,resultado,params)
    nombre_col = '{}_dias_{}_celdas_{}_splits'.format(dias,celdas,split)
    if not nombre_col in df.columns:
        df[nombre_col] = np.nan
        df[nombre_col + '_params'] = np.nan
    df.loc[alg,nombre_col] = resultado
    df.loc[alg,nombre_col + '_params'] = params
    df.to_pickle('resultados.pkl')
    
guarda_resultado('random_forest',1,2,0,3,'hola')
guarda_resultado('nearest_neighbor',1,2,0,4,'adios')
guarda_resultado('SVR',2,2,0,4,'salu2')
df 


# In[ ]:





# In[116]:


a = pd.read_pickle('resultados.pkl')
a


# In[ ]:


import os
from sklearn.model_selection import TimeSeriesSplit
import datetime

splits = [2,5,10]

res = ''
reinicia_df()
avistamientos = pd.read_pickle('../pkls/dfAvistamientos.pkl')
resultado = np.ravel(df_avistamientos)

listado_archivos = os.listdir('../pkls/')
df = pd.DataFrame(index=algoritmos)
for i in listado_archivos:
    for split in splits:
        tscv = TimeSeriesSplit(n_splits=split)
        if 'dfAtributosNormalizado' in i :
            atributos = pd.read_pickle('../pkls/{}'.format(i))
            dias,celdas = i[23:25],i[31:33]
            n_iter = 100
            print('\n\n' + i + '\n', flush=True)

            # concateno y guardo en log
            res += str(datetime.datetime.now()) + ' --------- ' +  str(i) + '\n'

            model_result = forest(atributos,resultado,n_iter,tscv)
            guarda_resultado('random_forest',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> Random Forest ' + str(model_result) + '\n'

            model_result = vecino(atributos,resultado,n_iter,tscv)
            guarda_resultado('nearest_neighbor',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> Vecino mas cercano ' + str(model_result) + '\n'

            model_result = vecino_grid(atributos,resultado,n_iter,tscv)
            guarda_resultado('nearest_neighbor_grid',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> Vecino mas cercano gridSearch ' + str(model_result) + '\n'

            model_result = SVR(atributos,resultado,n_iter,tscv)
            guarda_resultado('SVR',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> SVR ' + str(model_result) + '\n'

            model_result = SVR_grid(atributos,resultado,n_iter,tscv)
            guarda_resultado('SVR_grid',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> SVR_grid ' + str(model_result) + '\n'

            model_result = arbol_decision(atributos,resultado,n_iter,tscv)
            guarda_resultado('arbol_decision',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> arbol_decision ' + str(model_result) + '\n'

            model_result = arbol_decision_grid(atributos,resultado,n_iter,tscv)
            guarda_resultado('arbol_decision_grid',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> arbol_decision_grid ' + str(model_result) + '\n'
            
            model_result = boosting(atributos,resultado,n_iter,tscv)
            guarda_resultado('Boosting',dias,celdas,split,model_result[2],str(model_result[1]))
            res += '--> Boosting ' + str(model_result) + '\n'

        f = open ('log.txt','wb')

        f.write(bytes(res, encoding='utf-8'))
        f.close()


# In[97]:


## Regresion Lineal

df

