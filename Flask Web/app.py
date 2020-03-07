from flask import Flask,render_template

app = Flask(__name__)

@app.route('/')
def paginaPrincipal():
    return render_template('welcome.html')

@app.route('/pred')
def paginaPrediccion():
    return render_template('pred.html')

@app.route('/contacto')
def paginaContacto():
    return render_template('contacto.html')

if __name__ == '__main__':
    app.run(debug = True)


