import numpy as np
import flask
import tensorflow as tf
import tensorflow.keras as k
import cv2

def process_image(image):
    result = np.zeros_like(image)
    result[10:100, 10:100, 0] = 255
    return result

# https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf

def service():
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

    @app.route('/predict_cells')
    def predict_cells():
        image = np.zeros(shape=(320, 240, 3), dtype=np.uint8)
        image = process_image(image)
        _, encoded = cv2.imencode('.jpg', image)
        print(f'{type(encoded)=}')
        return flask.Response(response=encoded.tobytes(), mimetype='image/jpeg')


    app.run(host='0.0.0.0', port=9980)

def test_process_image():
    import matplotlib.pyplot as plt
    plt.clf()
    image = np.zeros(shape=(320, 240, 3), dtype=np.uint8)
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.subplot(2,2,2)
    plt.imshow(process_image(image))


if __name__ == "__main__":
    service()
    # test_process_image()
