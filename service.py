import numpy as np
import flask
import tensorflow as tf
import tensorflow.keras as k
import cv2
from resizable_autoencoder_model import load_resizable_autoencoder
from util.pad_image import pad_image


def process_image(image, resizable_autoencoder):
    image = image / 255.0
    full_model, expanded_image_shape = resizable_autoencoder.make_full_image_model(label_shape=image.shape)
    expanded_image = pad_image(image, shape=expanded_image_shape)
    result = full_model.predict(expanded_image[np.newaxis,...])
    result = result[0]
    # Visualize the result by showing only the borders.
    # TODO: port the mask extraction algorithm and use it here.
    result[..., 0] = 0 # Reset the red and blue channels
    result[..., 2] = 0
    result = np.clip(result, 0.0, 1.0)
    return ((image / 2.0 + result / 2.0) * 255.0).astype(np.uint8)


def service():
    resizable_autoencoder = load_resizable_autoencoder('resizable_autoencoder.h5')

    app = flask.Flask(__name__, static_url_path='', static_folder='static')
    @app.route('/')
    def home_endpoint():
        return app.send_static_file('./index.html')

    @app.route('/predict_cells', methods=['POST'])
    def predict_cells():
        file = flask.request.files['input_file']
        content = file.read()
        content = np.fromstring(content, dtype='uint8')
        print(f'{file=} {content=}')
        image = cv2.imdecode(content, cv2.IMREAD_COLOR)
        image = process_image(image, resizable_autoencoder)
        _, encoded = cv2.imencode('.jpg', image)
        print(f'{type(encoded)=}')
        return flask.Response(response=encoded.tobytes(), mimetype='image/jpeg')


    app.run(host='0.0.0.0', port=9980)

def test_process_image():
    from util.image_cache import image_cache
    import matplotlib.pyplot as plt

    resizable_autoencoder = load_resizable_autoencoder('resizable_autoencoder.h5')
    plt.clf()
    image = image_cache.get_data_image(2)
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.subplot(2,2,2)
    plt.imshow(process_image(image, resizable_autoencoder))
    plt.show()


if __name__ == "__main__":
    service()
    # test_process_image()
