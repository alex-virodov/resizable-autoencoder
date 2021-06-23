import numpy as np
import flask
import tensorflow as tf
import tensorflow.keras as k


# https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf

def main():
    model = k.models.load_model('iris.h5')
    model.summary()

    app = flask.Flask(__name__, static_url_path='', static_folder='static')
    @app.route('/')
    def home_endpoint():
        return app.send_static_file('./index.html')

    @app.route('/predict', methods=['POST'])
    def get_prediction():
        data = flask.request.form['data']
        print(f'{data=}')
        data = np.array([5.9, 3.0, 5.1, 1.8])
        print(f'{data=}')
        result = model.predict(data[np.newaxis, :])
        index = np.argmax(result[0], axis=0)
        print(f'{result=} {index=}')
        labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        return f'result={labels[index]} score={result[0, index]:0.6}'

    app.run(host='0.0.0.0', port=9980)



if __name__ == "__main__":
    main()
