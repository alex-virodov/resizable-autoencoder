from resizable_autoencoder import ResizableAutoencoder
import tensorflow as tf
import tensorflow.keras as k


def weight_edges_binary_crossentropy(y_true, y_pred):
    # Weight the edges more, as they are under-represented.
    weight_map = tf.where(y_true[..., 1] > 0.5, 2.0, 1.0)
    return tf.reduce_mean(
        tf.multiply(tf.keras.losses.binary_crossentropy(y_true, y_pred), weight_map))


def make_resizable_autoencoder() -> ResizableAutoencoder:
    # Capture the parameters so they can be shared between the train and eval functions
    return ResizableAutoencoder(n_folds=2, filter_size_schedule=[i*4 for i in [8,8,8,8]])


def load_resizable_autoencoder(model_file_name) -> ResizableAutoencoder:
    saved_model = k.models.load_model(
        model_file_name,
        custom_objects={
            'weight_edges_binary_crossentropy': weight_edges_binary_crossentropy
        })
    resizable_autoencoder = make_resizable_autoencoder()
    resizable_autoencoder.load_from_model(saved_model)
    return resizable_autoencoder
