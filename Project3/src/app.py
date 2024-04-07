from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def hello_world():
    return "Hello World"

@app.route('/<name>', methods = ['GET'])
def hello_name(name):
    return 'Hello, {}!'.format(name)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

