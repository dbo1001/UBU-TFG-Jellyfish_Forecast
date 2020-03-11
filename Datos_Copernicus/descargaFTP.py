import pandas as pd
import xarray
from ftplib import FTP
import os
import wget
from tqdm import tqdm
import sys

# Declaracion de variables
datos = dict()

def carga_datos():
    """
    Funcion que carga en una variable de tipo diccionario los datos necesarios para 
    descargar y filtrar los datos meteorologicos y oceanicos.
    """
    datos['FTP'] = 'my.cmems-du.eu' # direccion FTP
    datos['user'] = 'psantidriantuda' # usuario de la cuenta FTP
    datos['passwd'] = 'Kir@2110' # contrasena de la cuenta FTP
    datos['latitud'] = [-60, -10] # valores maximo y minimo de la latitud para las coordenadas GPS
    datos['longitud'] = [-120, -60] # valores maximo y minimo de la longitud para las coordenadas GPS
    datos['destino'] = os.getcwd() + '/descargas' #'C:/Users/pablo/Downloads' #'D:\##DatosCopernicus' # ruta en la que se descargaran los archivo ".nc"
    datos['fechas'] = pd.date_range(start='2014-01-01', end='2019-01-01', freq='Y') # rango de fechas a descargar
    datos['servicio'] = 'GLOBAL_REANALYSIS_PHY_001_030' # nombre de las carpetas contenedoras de los datos
    datos['producto'] = 'global-reanalysis-phy-001-030-daily'
    datos['nombre_inicio'] = 'mercatorglorys12v1_gl12_mean_' # nombre parcial de los documentos a descargar
    datos['variables'] = { # nombre las variables que se necesitan para el filtrado del fichero tras la descarga
        "bottomT":{
            "variable":"temperature_at_sea_floor",
            "extra_options":{}
        },
        "vo":{
                "variable":"northward_sea_water_velocity",
                # justo 0,5 y 10 no estan presentes, pero uso 'nearest'
                "extra_options": {"depth":[0,5,10]}
        },
        "uo":{
                "variable":"eastward_sea_water_velocity",
                "extra_options":  {"depth":[0,5,10]}
    
            
        },
        "mlotst":{        
                "variable":"ocean_mixed_layer_thickness",
                "extra_options":{}       
        },    
        "so":{
                "variable":"water_salinity",
                "extra_options":  {"depth":[0,5,10]}
        },    
        "zos":{
                "variable":"sea_surface_height",
                "extra_options":{}
            
        },    
        "thetao":{
                "variable":"temperature",
                "extra_options":  {"depth":[0,5,10]}
        }
    }

def acceder_FTP():
    '''
    Se realiza la conexion al servidor FTP
    '''
    global ftp
    ftp = FTP(host=datos['FTP'],user=datos['user'],passwd=datos['passwd'])
    ftp.cwd("Core" +'/'+ datos['servicio'] +'/'+ datos['producto'])

    
def comprobar_fichero(fichero):
    '''
    Comprueba la existencia del fichero en local.

    Recibe:
        - fichero --> Nombre del archivo a comprobar
    Devuelve:
        - False --> En caso de no existir.
        - True  --> En caso de existir.
    '''
    ruta = datos['destino']
    if fichero in os.listdir(ruta): 
        print('Fichero: "{}" ya existe en local.'.format(fichero))
        return True
    return False

def lanza_comando(dia,tamano):
    '''
    Se lanza el comando de descarga.

    Recibe:
        - dia --> dia que se quiere descargar
        - tamano --> tamano del fichero
    '''
    p = dia
    with open(datos['destino'] + '/' + p, 'wb') as fd:
        total = ftp.size(p)
        with tqdm(total=total,
                unit_scale=True,
                desc=p,
                miniters=1,
                file=sys.stdout,
                leave=True
                ) as pbar:
            def cb(data):
                pbar.update(len(data))
                fd.write(data)
            ftp.retrbinary('RETR {}'.format(p), cb)
        fd.close()

def crop_datos(data):
    """
    Recorta la zona correspondiente a la costa de chile

    Recibe:
        - data --> fichero a recortar

    Devuelve:
        - croped_data --> fichero ya recortado
    """
    lat_bnds = datos['latitud']
    lon_bnds = datos['longitud']
    croped_data = data.sel(latitude=slice(*lat_bnds), 
                           longitude=slice(*lon_bnds))
    return croped_data

def filtrar_datos(data,options_data):
    """
    Elimina todas aquellas variables que no queremos
    Elimina todas las profundidades excepto 3.
    
    Recibe:
        - data --> archivo inicial 
        - options_data --> diccionario con las variables
    Devuelve:
        - archivo filtrado con las varaibes deseadas
    """
    filtered_data = data.copy()

    for var in list(filtered_data.keys()):
        if not var in options_data:
            #print("Deleting not necesary var",var)
            del filtered_data[var]
    # Selecciona 3 profundidades mas cercanas a las indicadas    
    return filtered_data.sel({"depth":[0,5,10]},method='nearest')

    
def tratar_fichero(fichero):
    """
    LLama a las funciones de recorte y eliminacion de las variables no deseadas
    Recibe:
        - fichero --> archivo inicial 
    Devuelve:
        - datos_filtrados --> archivo tratado
    """
    datos_brutos = xarray.open_dataset(datos['destino'] + '/' + fichero)
    datos_crop = crop_datos(datos_brutos)
    datos_filtrados = filtrar_datos(datos_crop,datos['variables'])
    return datos_filtrados
    
     
def procesar():
    '''
    Recorre las carpetas del FTP entre las fechas requeridas
    '''
    global ftp, tamano
    for anio in datos['fechas'].year:
        ftp.cwd(str(anio))
        for mes in ftp.nlst():
            ftp.cwd(str(mes))
            for dia in ftp.nlst():
                #Comprobar que esta descargado
                descargado = comprobar_fichero(dia)
                #Descargar
                if not descargado:
                    tamano = ftp.size(dia)
                    print(' --> Descargando - Ano: {} / Mes: {} / Dia: {} - Nombre: {}'.format(anio,mes,str(dia)[35:37],dia))
                    lanza_comando(dia,tamano)
                    datos_nuevos = tratar_fichero(dia)
                    datos_nuevos.to_netcdf(datos['destino'] +'/'+ dia.replace('.nc','__filtrados.nc'))
                    print(' --> Guardado fichero modificado y borramos el anterior.\n')
                    os.remove(datos['destino'] +'/'+dia)
                    # os.unlink(datos['destino'] +'\\'+dia)
                    
            ftp.cwd("../")
        ftp.cwd("../")
    

if __name__ == "__main__":
    carga_datos()
    acceder_FTP()
    procesar()


