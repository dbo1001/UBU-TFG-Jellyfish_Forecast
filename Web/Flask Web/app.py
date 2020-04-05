from flask import Flask,render_template
from flask_bootstrap import Bootstrap
import folium
import os
import json
import pandas as pd


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def paginaPrincipal():
    return render_template('index.html')

@app.route('/mapas')
def paginaPrediccion():
    start_coords = (-34.536267, -72.406639)
    folium_map = folium.Map(location=start_coords, zoom_start=5)
    print(os.getcwd())
    print('')
    folium_map.save('web\\Flask Web\\templates\\mapaChile.html')
    return render_template('mapas.html')

@app.route('/contacto')
def paginaContacto():
    return render_template('contacto.html')


@app.route('/pruebas')
def index():
    start_coords = (-34.536267, -72.406639)
    folium_map = folium.Map(location=start_coords, zoom_start=5)
    gjson = os.path.join(app.root_path, 'geojson/prueba.geojson')
    file = open(gjson,encoding='utf-8')
    geo_data = json.load(file)
    file.close()
    print(geo_data)
    #imprimir playas
    w = geo_data['geometry']
    df = pd.read_excel('web\\Flask Web\\geojson\\playas.xlsx')
    print(df)
    #for x,y in enumerate(w['coordjuinates']):
    for x,y in df.iterrows():
        #print('-----------------------------------',x,y)
        folium.Marker(
            #location=[y[0],y[1]],
            #popup=w["nombre"][x]
            location = [str(df.loc[x]['Latitud']).replace(',','.'),str(df.loc[x]['Longitud']).replace(',','.')],
            popup=str(df.loc[x]['Latitud']).replace(',','.') +''+''+ str(df.loc[x]['Longitud']).replace(',','.')
        ).add_to(folium_map)
        
    folium_map.save('web\\Flask Web\\templates\\mapaChile.html')
    return render_template('mapas.html')

if __name__ == '__main__':
    app.run(debug = True)


