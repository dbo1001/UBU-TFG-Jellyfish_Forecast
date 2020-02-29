from flask import Flask,render_template

app = Flask(__name__)

@app.route('/')
def paginaPrincipal():
    return render_template('welcome.html')

@app.route('/about')
def paginaAbout():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug = True)


