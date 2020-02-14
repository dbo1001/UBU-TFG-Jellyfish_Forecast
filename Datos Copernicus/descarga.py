import pandas as pd
import subprocess
import os

'''
¿Todos los miércoles?
'''

def carga_datos():  
    global datos 
    datos = dict()
    datos['user'] = 'psantidriantuda'
    datos['passwd'] = 'Kir@2110' 
    datos['latitud']  = ['-60','-10']
    datos['longitud']  = ['-120','-60']
    datos['destino'] = os.getcwd() + '\\Datos Copernicus\\Datos'
    datos['fechas'] = ['2014-01-01 00:00:00' ,'2019-01-01 00:00:00']
    datos['servicio'] = 'GLOBAL_REANALYSIS_PHY_001_030-TDS'
    datos['producto'] = 'global-reanalysis-phy-001-030-daily'
    
def rellenar_comando():
    comando = "python -m motuclient --motu http://nrt.cmems-du.eu/motu-web/Motu --service-id "+ datos['servicio'] +" --product-id "+ datos['producto'] +" "\
                "--longitude-min "+ datos['longitud'][0] +" --longitude-max "+ datos['longitud'][1] +" --latitude-min "+ datos['latitud'][0] +" --latitude-max "+ datos['latitud'][1] +" "\
                "--date-min "+ datos['fechas'][0] +" --date-max "+ datos['fechas'][1] +" --depth-min 0.493 --depth-max 0.4942 --variable thetao --variable bottomT --variable so "\
                "--variable zos --variable uo --variable vo --variable mlotst --variable siconc --variable sithick --variable usi --variable vsi --out-dir "\
                '"'+ datos['destino'] + '"' +" --out-name "+ datos['producto'] +" --user "+ datos['user'] +" --pwd "+ datos['passwd'] 
    print (comando)
    return comando

def bajar_ficheros():
    comando = rellenar_comando()
    proceso1 = subprocess.Popen(comando)
    proceso1.wait()

if __name__ == "__main__":
    carga_datos()
    bajar_ficheros()



#python -m motuclient --motu http://my.cmems-du.eu/motu-web/Motu --service-id GLOBAL_REANALYSIS_PHY_001_030-TDS --product-id global-reanalysis-phy-001-030-daily --longitude-min -180 --longitude-max 179.9166717529297 --latitude-min -80 --latitude-max 90 --date-min "2018-12-01 12:00:00" --date-max "2018-12-03 12:00:00" --depth-min 0.493 --depth-max 0.4942 --variable thetao --variable bottomT --variable so --variable zos --variable uo --variable vo --variable mlotst --variable siconc --variable sithick --variable usi --variable vsi --out-dir C:\Users\pablo\Desktop --out-name prueba.nc --user psantidriantuda --pwd Kir@2110