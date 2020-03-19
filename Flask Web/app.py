from flask import Flask,render_template
from flask_bootstrap import Bootstrap
import folium

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def paginaPrincipal():
    return render_template('index.html')

@app.route('/mapas')
def paginaPrediccion():
    return render_template('mapas.html')

@app.route('/contacto')
def paginaContacto():
    return render_template('contacto.html')


@app.route('/pruebas')
def index():
    start_coords = (-34.536267, -72.406639)
    folium_map = folium.Map(location=start_coords, zoom_start=5)
    folium_map.save('templates/mapita.html')
    return render_template('mapas.html')

if __name__ == '__main__':
    app.run(debug = True)


