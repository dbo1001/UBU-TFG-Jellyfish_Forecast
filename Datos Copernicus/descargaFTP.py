import pandas as pd
import xarray
from ftplib import FTP
import os
from datetime import datetime
from subprocess import Popen, PIPE
import wget
from tqdm import tqdm
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer, UnknownLength

# MEJORAS -->   pedir por pantalla el lugar de descarga de los archvos
#               cambio progress bar por libreria clint (https://likegeeks.com/es/descargar-archivos-usando-python/#Descargar-con-una-barra-de-progreso)

# Declaración de variables
datos = dict()
pbar = ProgressBar()

def carga_datos():
    """
    Función que carga en una variable de tipo diccionario los datos necesarios para 
    descargar y filtrar los datos meteorológicos y oceánicos.
    """
    datos['FTP'] = 'my.cmems-du.eu' # dirección FTP
    datos['user'] = 'psantidriantuda' # usuario de la cuenta FTP
    datos['passwd'] = 'Kir@2110' # contraseña de la cuenta FTP
    datos['latitud'] = ['-60', '-10'] # valores máximo y mínimo de la latitud para las coordenadas GPS
    datos['longitud'] = ['-120', '-60'] # valores máximo y mínimo de la longitud para las coordenadas GPS
    datos['destino'] = 'C:/Users/pablo/Downloads' #'D:\##DatosCopernicus' # ruta en la que se descargarán los archivo ".nc"
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
                # justo 0,5 y 10 no están presentes, pero uso 'nearest'
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
    Se realiza la conexión al servidor FTP
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
        - dia --> día que se quiere descargar
        - tamano --> tamaño del fichero

    '''
    archivo = open(datos['destino'] + '\\' + dia,'wb')
    widgets = ['Downloading: ', Percentage(), ' ',
                    Bar(marker='#',left='[',right=']'),
                    ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=tamano)
    pbar.start()
    def file_write(data):
        archivo.write(data) 
        # global pbar
        # pbar += len(data)
    
    ftp.retrbinary('RETR '+ dia ,file_write)
    archivo.close()
    print('Fin descarga.')

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
    """
    filtered_data = data.copy()

    for var in list(filtered_data.keys()):
        if not var in options_data:
            #print("Deleting not necesary var",var)
            del filtered_data[var]
    # Selecciona 3 profundidades más cercanas a las indicadas    
    return filtered_data.sel({"depth":[0,5,10]},method='nearest')

    
def tratar_fichero(fichero):
    datos_brutos = xarray.open_dataset(datos['destino'] + '\\' + fichero)
    datos_crop = crop_datos(datos_brutos)
    datos_filtrados = filtrar_datos(datos_crop,datos['variables'])
    return datos_filtrados
    
     
def procesar():
    global ftp, tamano
    for año in datos['fechas'].year:
        ftp.cwd(str(año))
        for mes in ftp.nlst():
            ftp.cwd(str(mes))
            for dia in ftp.nlst():
                #Comprobar que esta descargado
                descargado = comprobar_fichero(dia)
                #Descargar
                if not descargado:
                    tamano = ftp.size(dia)
                    print("Descargando {} // {}".format(dia,tamano))
                    lanza_comando(dia,tamano)
                    datos_nuevos = tratar_fichero(dia)
                    datos_nuevos.to_netcdf(datos['destino'] +'\\'+ dia.replace('.nc','__filtrados.nc'))
                    print('\nGuardado fichero modificado y borramos el anterior.')
                    # os.remove(datos['destino'] +'\\'+dia)
            ftp.cwd("../")
        ftp.cwd("../")
    

if __name__ == "__main__":
    carga_datos()
    acceder_FTP()
    procesar()


