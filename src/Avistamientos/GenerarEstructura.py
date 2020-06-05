#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from math import floor
import math
import os
from datetime import datetime
import xarray as xr
from tqdm.notebook import tqdm
from sklearn import preprocessing
from time import time


# In[2]:


def guarda_dataframe(df,nombre,subcarpeta = False):
    if subcarpeta:
        df.to_pickle('./pkls/{}.pkl'.format(nombre))
        df.to_excel('./Excels/{}.xlsx'.format(nombre))
    else:
        df.to_pickle('{}.pkl'.format(nombre))
        df.to_excel('{}.xlsx'.format(nombre))
    


# # Listado de playas con avistamientos y fechas

# ## con primer grupo de datos

# In[3]:


avistamientos_df = pd.read_excel("./ExcelsAvistamientosIniciales/Physalia_Ambiental_R.xlsx")
# avistamientos_df = pd.read_excel("../Physala_Data/Datos_Physalia_20171010.xls")

columnas = avistamientos_df.iloc[0]

#Quito las 3 primeras filas, debido al formato de la excell
avistamientos_df = avistamientos_df.iloc[3:] 
avistamientos_df.columns = columnas

# Me quedo solo con los datos de avistamientos
avistamientos_df = avistamientos_df[["Latitud","Longitud","Año","Mes","Día","Avistamientos"]]

#Transdormaciones para sacar con fecha (datetime)
avistamientos_fecha_df=avistamientos_df[["Año","Mes","Día"]]
avistamientos_fecha_df.columns = ["year","month","day"]

fecha = pd.to_datetime(avistamientos_fecha_df)

avistamientos_df["Fecha"]=fecha
avistamientos_df=avistamientos_df[["Latitud","Longitud","Fecha","Avistamientos"]]
avistamientos_df


# ## con segundo grupo de datos

# In[4]:


avistamientos_2_df = pd.read_excel("./ExcelsAvistamientosIniciales/Datos_Physalia_20171010.xls")

# elimino filas con tipo de atributo "Categoria" que van del 1 al tres y no siguen la mismas reglas que los numericos
avistamientos_2_df = avistamientos_2_df[avistamientos_2_df['Tipo.Abund'] == 'numero']
# cambio signo de coordenadas 
avistamientos_2_df['Lat.dec'] = avistamientos_2_df['Lat.dec'].multiply(-1)
avistamientos_2_df['Long.dec'] = avistamientos_2_df['Long.dec'].multiply(-1)

#filtro las columnas que nos interesan
avistamientos_2_df = avistamientos_2_df[['Lat.dec','Long.dec','Date','Abundancia']]
avistamientos_2_df.columns = ['Latitud','Longitud','Fecha','Avistamientos']


# In[5]:


#filas sin datos
filas_nan = avistamientos_2_df[avistamientos_2_df.isna().any(axis=1)] # filas con valores vacios
print(filas_nan)
# eliminar esas filas
avistamientos_2_df = avistamientos_2_df.drop(filas_nan.index.values)


# In[6]:


#guarda df
guarda_dataframe(avistamientos_2_df,'1_avistamientos_origen')

avistamientos_2_df


# In[7]:


## para leer los .pkl
# a  = pd.read_pickle('1_avistamientos.pkl')


# # Redondeo con salto entre cuadrantes de dataSet original (0,0833)
# Redondeo de latitud y longitud para juntar lecturas de una misma playa con coordenadas muy proximas
# 
# Se exporta el dataframe generado a un excel

# In[8]:


base = 1/4
def redondeo(x,y):
    return  [base * round(x/base),base * round(y/base)]
x = redondeo(37.1707222222222,-73.2053333333333)
x


# In[9]:


df_avistamientos  = pd.read_pickle('1_avistamientos_origen.pkl')


# In[10]:


base=1/12

def redondeo(x):
    return  base * round(x/base)

#Se añaden atributos con la longitud y latitud redondeadas a cada cuarto de grado
df_avistamientos["Lat_floor"] = df_avistamientos.Latitud.map(redondeo)
df_avistamientos["Long_floor"] = df_avistamientos.Longitud.map(redondeo)

#Descarto latitudes sin redondear
df = df_avistamientos[['Lat_floor', 'Long_floor',"Fecha","Avistamientos"]]

#DataFrame total avistamientos de playas en esa cuadricula
df_sum = df.groupby(['Lat_floor', 'Long_floor',"Fecha"]).sum()

#DataFrame número de playas en esa cuadricula
df_count = df.groupby(['Lat_floor', 'Long_floor',"Fecha"]).count()

#Dataframe con el total de avistamientos y el número de playas
df_join = df_sum.join(df_count,lsuffix="I",rsuffix="R")
df_join.columns=['Suma', 'N_Playas']
df_join['Media_Playas'] = df_join.Suma /df_join.N_Playas

#Se ordenan temporalmente
df_join = df_join.sort_values(by=['Fecha','Lat_floor','Long_floor' ]).reset_index().set_index(['Fecha','Lat_floor','Long_floor' ])

# DataFrame de avistamientos procesado
guarda_dataframe(df_join,'2_avistamientos_redondeo')


# In[11]:


df_join


# # Estructura uniendo avistamientos y condiciones oceánicas

# In[12]:


df_join = pd.read_pickle('2_avistamientos_redondeo.pkl')


# In[13]:


'''
Las celdas que estaban unidas por tener el mismo valor, tienen valores nulos
por lo que las relleno con us valor correspodiente
'''
df_playas = df_join.reset_index()
# Cojo las columnas que me interesan
df_playas = df_playas[["Suma","Lat_floor","Long_floor","Fecha"]]
# Renombro las columnas 
df_playas = df_playas.rename(columns={"Suma": "Avistamientos", "Lat_floor": "Latitud", "Long_floor": "Longitud"})
# Ordeno por fecha
df_playas = df_playas.sort_values(by=["Fecha"])
guarda_dataframe(df=df_playas,nombre='2_avistamientos_redondeo')


# ### características

# In[14]:


df_playas # DataFrame con avistamientos en cada playa
cuadrantes = { # Numero de cuadrantes que se concatenarán con las playas
    'izquierda': 0 ,
    'arriba': 0 ,
    'abajo': 0
}
dias_desfase = 15
salto = 1/12 # Salto de coordenadas entre los cuadrantes
listado_archivos = os.listdir('../../descargas') # Listo todos los archivos de Copernicus
variables = ['Coord','Profundidad','mlotst','zos','bottomT','thetao','so','uo','vo']


# In[15]:


def busca_archivo(fecha):
    texto ='_{}_'.format(str(fecha).split()[0].replace('-',''))
    archivo = [x for x in listado_archivos if str(texto) in x]
    data = xr.open_dataset('../../descargas/{}'.format(archivo[0])) # cargo el archivo
    return data # devuelvo dataset

def dame_datos(latitud,longitud,ds):
    return ds.sel({'latitude':latitud,'longitude': longitud})

def comprueba_datos(latitud,longitud,ds):
    valor = dame_datos(latitud,longitud,ds)
    while math.isnan(valor.mlotst[0]):
        longitud = longitud - salto
        valor = dame_datos(latitud,longitud,ds)
    return latitud,longitud # devuelvo las coordenadas con datos

def crea_DF():
    df_resultados = None
    df_resultados = pd.DataFrame(columns=["Avistamientos","Latitud","Longitud","Fecha",'Coord',"Profundidad","mlotst",'zos','bottomT','thetao','so','uo','vo']) # columnas básicas
    for izquierda in range(cuadrantes['izquierda']): # por cada cuadrante a la izquierda meto un grupo de columnas
        var_aux = [str(x+'_izq_'+str(izquierda+1)) for x in variables]
        df_aux = pd.DataFrame(columns=var_aux)
        df_resultados = df_resultados.join(df_aux)
        for superior in range (cuadrantes['arriba']): # por cada cuadrante a la superior meto un grupo de columnas
            var_aux = [x+'_sup_'+str(izquierda+1)+'.'+str(superior+1) for x in variables]
            df_aux = pd.DataFrame(columns=var_aux)
            df_resultados = df_resultados.join(df_aux)
        for inferior in range (cuadrantes['abajo']): # por cada cuadrante a la inferior meto un grupo de columnas    
            var_aux = [x+'_inf_'+str(izquierda+1)+'.'+str(inferior+1) for x in variables]
            df_aux = pd.DataFrame(columns=var_aux)
            df_resultados = df_resultados.join(df_aux)
    return df_resultados

def meto_en_df(df_resultados,row,coordenadas,df,indice,variables_actuales):
    for x in range(3):
        df_resultados.loc[indice + x,["Avistamientos","Latitud","Longitud","Fecha"]] = row.values.tolist() # meto en df_resulados los datos de ese cuadrante
        vars = df.loc[x,['depth', 'mlotst','zos','bottomT','thetao','so','uo','vo']].values.tolist()
        vars.insert(0,coordenadas)
        df_resultados.loc[indice + x,variables_actuales] = vars # meto coordenadas reales utilizadas y datos 
    return df_resultados

def dameCoordIzq(lat,long,n_saltos):
    return [lat,long-(salto*n_saltos)]

def dameCoordSuperior(lat,long,n_saltos):
    coordenadas = dameCoordIzq(lat,long,n_saltos)
    return [lat+(salto*n_saltos),long]

def dameCoordInferior(lat,long,n_saltos):
    return [lat-(salto*n_saltos),long]

def resta_fecha(fecha,dias):
    return fecha - pd.Timedelta(days=2)


# In[16]:


def crea_estr(df):
    df_resultados = crea_DF() # creo df con columnas necesarias
    #print(df_resultados)
    total_lineas = df.shape[0]
    pbar = tqdm(total = total_lineas)
    for contador,[index, row] in enumerate(df.iterrows()): # por cada playa
#         print(contador,end='\r')
        pbar.update(1)
        ds = busca_archivo(resta_fecha(row['Fecha'],dias_desfase)) # busco el archivo de la fecha 
        coordenadas = comprueba_datos(row['Latitud'],row['Longitud'],ds) # miro a ver si el cuadrante tiene datos, sino cojo el siguiente por la izquierda

        df = dame_datos(coordenadas[0],coordenadas[1],ds).to_dataframe().reset_index() # dataframe con los datso oceanicos de esas coordenadas y en esa fecha
        df_resultados = meto_en_df(df_resultados,row,list(coordenadas),df,contador * 3,variables)
#         print('--{}'.format(contador))
        # datos de las celdas extras
        for izquierda in range(cuadrantes['izquierda']): # por cada cuadrante a la izquierda, añado datos
            coordenadas = dameCoordIzq(coordenadas[0],coordenadas[1],izquierda+1)
            df = dame_datos(coordenadas[0],coordenadas[1],ds).to_dataframe().reset_index() 
            var_aux = [x+'_izq_'+str(izquierda+1) for x in variables]
            df_resultados = meto_en_df(df_resultados,row,coordenadas,df,contador * 3,var_aux)
            coord_aux = coordenadas
            for superior in range (cuadrantes['arriba']): # por cada cuadrante a la superior, añado datos
                coord_aux = dameCoordSuperior(coord_aux[0],coord_aux[1],superior+1)
                df = dame_datos(coord_aux[0],coord_aux[1],ds).to_dataframe().reset_index() 
                var_aux = [x+'_sup_'+str(izquierda+1)+'.'+str(superior+1) for x in variables]
                df_resultados = meto_en_df(df_resultados,row,coord_aux,df,contador * 3,var_aux)
            coord_aux = coordenadas
            for inferior in range (cuadrantes['abajo']): # por cada cuadrante a la inferior, añado datos  
                coord_aux = dameCoordInferior(coord_aux[0],coord_aux[1],inferior+1)
                df = dame_datos(coord_aux[0],coord_aux[1],ds).to_dataframe().reset_index() 
                var_aux = [x+'_inf_'+str(izquierda+1)+'.'+str(inferior+1) for x in variables]
                df_resultados = meto_en_df(df_resultados,row,coord_aux,df,contador * 3,var_aux)

#         if contador == 5:
#             break
    return df_resultados


# In[17]:


# creo estructura de datos
df_resultados = crea_estr(df_playas)
df_resultados.head()


# In[18]:


df_resultados.head()


# In[19]:


# #Guardo la estructura
# df_resultados = df_resultados.reset_index().set_index(['Fecha','Latitud','Longitud','Avistamientos','Profundidad'])
# guarda_dataframe(df_resultados,nombre='3_estructura')
# df_resultados.to_excel('3EstructuraFinal.xlsx')
# df_resultados.head()


# In[20]:


# for i in [0,7,14,30,45,60]:
#     dias_desfase = i
#     df_resultados = crea_estr()
#     df_resultados = df_resultados.reset_index().set_index(['Latitud','Longitud','Fecha','Avistamientos','Profundidad'])
#     df_resultados.to_excel('-EstructuraFinal{}dias.xlsx'.format(i))
# df_resultados.head()


# ## Estructura sin rellenar huecos sin datos

# In[21]:


# def crea_estr():
#     df_resultados = crea_DF() # creo df con columnas necesarias
#     #print(df_resultados)
#     total_lineas = df_playas.shape[0]
#     pbar = tqdm(total = total_lineas)
#     for contador,[index, row] in enumerate(df_playas.iterrows()): # por cada playa
# #         print(contador,end='\r')
#         pbar.update(1)
#         ds = busca_archivo(row['Fecha']) # busco el archivo de la fecha 
#         coordenadas = [row['Latitud'],row['Longitud']]

#         df = dame_datos(row['Latitud'],row['Longitud'],ds).to_dataframe().reset_index() # dataframe con los datso oceanicos de esas coordenadas y en esa fecha
#         df_resultados = meto_en_df(df_resultados,row,[row['Latitud'],row['Longitud']],df,contador * 3,variables)
        
#         # datos de las celdas extras
#         for izquierda in range(cuadrantes['izquierda']): # por cada cuadrante a la izquierda, añado datos
#             coordenadas = dameCoordIzq(coordenadas[0],coordenadas[1],izquierda+1)
#             df = dame_datos(coordenadas[0],coordenadas[1],ds).to_dataframe().reset_index() 
#             var_aux = [x+'_izq_'+str(izquierda+1) for x in variables]
#             df_resultados = meto_en_df(df_resultados,row,coordenadas,df,contador * 3,var_aux)
#             coord_aux = coordenadas
#             for superior in range (cuadrantes['arriba']): # por cada cuadrante a la superior, añado datos
#                 coord_aux = dameCoordSuperior(coord_aux[0],coord_aux[1],superior+1)
#                 df = dame_datos(coord_aux[0],coord_aux[1],ds).to_dataframe().reset_index() 
#                 var_aux = [x+'_sup_'+str(izquierda+1)+'.'+str(superior+1) for x in variables]
#                 df_resultados = meto_en_df(df_resultados,row,coord_aux,df,contador * 3,var_aux)
#             coord_aux = coordenadas
#             for inferior in range (cuadrantes['abajo']): # por cada cuadrante a la inferior, añado datos  
#                 coord_aux = dameCoordInferior(coord_aux[0],coord_aux[1],inferior+1)
#                 df = dame_datos(coord_aux[0],coord_aux[1],ds).to_dataframe().reset_index() 
#                 var_aux = [x+'_inf_'+str(izquierda+1)+'.'+str(inferior+1) for x in variables]
#                 df_resultados = meto_en_df(df_resultados,row,coord_aux,df,contador * 3,var_aux)
        
#     return df_resultados


# In[22]:


# df_resultados_con_vacios = crea_estr()


# In[23]:


# df_resultados_con_vacios.head()


# In[24]:


# df_resultados_con_vacios.to_excel('3EstructuraCoordEnTierra.xlsx')


# # Rellenar datos de celdas vacios
# Con datos de la coordenada más cercana por la izquierda

# In[25]:


def comprueba_datos_thetao(latitud,longitud,ds):
    valor = dame_datos(latitud,longitud,ds)
    while math.isnan(valor.thetao[0][2]):
        longitud = float(longitud) - salto
        valor = dame_datos(latitud,longitud,ds)
    return latitud,longitud # devuelvo las coordenadas con datos


# In[26]:


# datos_excel = pd.read_excel('EstructuraFinal.xlsx').reset_index() # leo estructura de datos que contiene huecos sin informacion
# datos = datos_excel.drop(['Unnamed: 0'],axis=1).sort_values(by=['Latitud','Longitud','Profundidad','Fecha']) 

# datos_excel = df_resultados.copy().reset_index()
# datos = datos_excel.sort_values(by=['Latitud','Longitud','Fecha']) 


# In[27]:


# datos


# In[28]:


def rellenar_huecos(df):
    filas_nan = df[df.isna().any(axis=1)] # filas con valores vacios
    datos_copia = df.copy()
    total_lineas = filas_nan.shape[0]
    pbar = tqdm(total = total_lineas)
    for indice, fila in filas_nan.iterrows():
#         print(indice,fila)
        pbar.update(1)
        elem = fila.isna() # columnas con valores nan
        for index, value in elem.items():
            if value:
                nombre_original= index
                nombre = 'Coord'
                try:
                    nombre_original = index[:index.index('_')] # nombre de la columna en el DataSet
                    nombre = nombre + index[index.index('_'):] # nombre de la columna de coordenadas de ese punto
                except:
                    pass
                coordenadas = fila[nombre]
#                 coordenadas = [0,0]
#                 coordenadas[0] = coord[1:coord.index(',')]
#                 coordenadas[1] = coord[coord.index(',')+1:coord.index(']')]
                #print(fila)
                archivo = busca_archivo(fila['Fecha'])
                coordenadas = comprueba_datos_thetao(coordenadas[0],coordenadas[1],archivo)
                datos_fila = dame_datos(coordenadas[0],coordenadas[1],archivo)
                valor =datos_fila.sel({'depth':fila['Profundidad']}).to_dataframe().reset_index()[nombre_original].values[0]
                datos_copia.loc[indice,index] = valor
    return datos_copia


# In[29]:


# datos_sin_huecos = rellenar_huecos(datos)


# In[30]:


# filas_nan_2 = datos_sin_huecos[datos_sin_huecos.isna().any(axis=1)] # compruebo que se hayan rellenado todos los vacios
# filas_nan_2.head()


# In[31]:


# datos_sin_huecos = datos_sin_huecos.reset_index().set_index(['Latitud','Longitud','Fecha','Avistamientos','Profundidad']).drop(['index'],axis = 1)
# datos_sin_huecos.to_excel('4EstructuraFinalSinMissings.xlsx')


# In[32]:


# datos_sin_huecos


# ## Ajustes estructura + normalización

# In[33]:


# Estructura quitando las columnas de las coordenadas pues no nos sirven
def cargar(nombre):
    df = pd.read_pickle(nombre)
    df = df.fillna(method='ffill', axis=0).set_index(['Latitud','Longitud','Fecha','Avistamientos','Profundidad'])#.drop(['level_0','index'],axis = 1)
    # Eliminar columnas Coords
    cols = [c for c in df.columns if 'Coord' not in c and 'Profundidad' not in c]

    df_inicial=df[cols]

    return df_inicial


# In[34]:


profundidades = ['_0','_5','_10']

# añado etiqueta de profundidad
def renombra_atributos(df):
    columnas = []
    for i in range(3):
        columnas_aux = df.columns + profundidades[i]
        [columnas.append(x) for x in columnas_aux]
    df_atributos = pd.DataFrame(columns=columnas)
    return df_atributos
# df_atributos


# In[35]:


def unifica_lineas(df):
    df_atributos = renombra_atributos(df)
    # meto tres profundidades consecutivas
    total_lineas = df.shape[0]/3
    pbar = tqdm(total = total_lineas)
    for cont in range(int(df.shape[0]/3)):
        pbar.update(1)
        inicio = cont*3
        df_aux = df.iloc[inicio:inicio+3]
        lista = list()

        for fila in df_aux.values.tolist():
            for elem in fila :
                lista.append(elem)

        dict_atr = dict(zip(df_atributos.columns,lista))
        df_atr_aux = pd.DataFrame(dict_atr,index=[0])
        df_atributos = pd.concat([df_atributos,df_atr_aux])
    pbar.update(0)
    return df_atributos


# In[36]:


# def normaliza_normalize(df_atributos):
#     X = df_atributos.values.tolist()
#     x_normalizado = preprocessing.normalize(X, norm='l2')

#     x_normalizado

#     df_norm = pd.DataFrame(x_normalizado,columns=df_atributos.columns)
#     df_norm.to_excel('dfAtributosNormalizado.xlsx')
#     guarda_dataframe(df_norm,'dfAtributosNormalizado_{}dias_{}celdas.xlsx'.format(dias_desfase,cuadrantes['izquierda']))


# In[41]:


def normaliza_min_max(df_atributos,nombre,dias,celdas):
    X = df_atributos.values.tolist()
    min_max = preprocessing.MinMaxScaler()
    x_normalizado_2 = min_max.fit_transform(X)
    x_normalizado_2
    df_norm = pd.DataFrame(x_normalizado_2,columns=df_atributos.columns)
#     df_norm.to_excel('dfAtributosNormalizado2_{}.xlsx'.format(nombre[3:]))
    guarda_dataframe(df_norm,'dfAtributosNormalizado_{}_dias_{}_celdas'.format(dias,celdas),True)
    return df_norm


# In[38]:


def lista_avistamientos(nombre):
    df = pd.read_pickle(nombre)
    df = df.fillna(method='ffill', axis=0)#.drop(['level_0','index'],axis = 1)
    listado_avistamientos = df.Avistamientos.values.tolist()
    avistamientos = [x for x in listado_avistamientos[::3]]
    df_avistamientos = pd.DataFrame(avistamientos,columns=['Avistamientos'])
    guarda_dataframe(df_avistamientos,'dfAvistamientos',True)
    return df_avistamientos


# In[49]:


# coloca las tres profundidades en una misma linea
def ejecuta(nombre,dias,celdas):
    df_inicial = cargar(nombre)
    df_atributos = unifica_lineas(df_inicial)
    df_atri_norm = normaliza_min_max(df_atributos,nombre,dias,celdas)
    df_avis_norm = lista_avistamientos(nombre)
    return df_atri_norm,df_avis_norm


# # Proceso completo con diferentes configuraciones

# In[50]:


configs = {'num_dias':[0,7,15,30,45,60],
           'celdas':[0,1,2,3,4,5]}

df = pd.read_pickle('2_avistamientos_redondeo.pkl')
inicio = time()
for d in configs['num_dias']:
    dias_desfase = d # dias de desfase respecto la fecha de avistamiento
    for c in configs['celdas']:
        print(time() - inicio)
        cuadrantes['izquierda'],cuadrantes['arriba'],cuadrantes['abajo'] = c,c,c
        print('{}dias {}celdas'.format(d,c))
        df_res = crea_estr(df)
        df_res = df_res.sort_values(by=['Latitud','Longitud','Fecha'])
#         print(df_res)
        df_res = rellenar_huecos(df_res)
        guarda_dataframe(df_res,'3estruct_{}dias_{}celdas'.format(d,c),True)
        ejecuta(nombre='./pkls/3estruct_{}dias_{}celdas.pkl'.format(d,c),dias=d,celdas=c)
        
        


# In[ ]:




