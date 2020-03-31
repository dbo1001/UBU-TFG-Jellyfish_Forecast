from flask import Flask,render_template
from flask_bootstrap import Bootstrap
import folium
import os
import json


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def paginaPrincipal():
    return render_template('index.html')

@app.route('/mapas')
def paginaPrediccion():
    start_coords = (-34.536267, -72.406639)
    folium_map = folium.Map(location=start_coords, zoom_start=5)
    folium_map.save('Flask Web\\templates\\mapaChile.html')
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
    
    w = geo_data['geometry']
    for x,y in enumerate(w['coordinates']):
        print('-----------------------------------',x,y)
        folium.Marker(
            location=[y[0],y[1]],
            popup=w["nombre"][x]
        ).add_to(folium_map)
        
    folium_map.save('Flask Web\\templates\\mapaChile.html')
    return render_template('mapas.html')

if __name__ == '__main__':
    app.run(debug = True)


