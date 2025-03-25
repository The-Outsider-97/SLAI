from flask import Flask, render_template

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/styles')

@app.route('/')
def index():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)
