import tensorflow.keras as k
import math
import unittest
from typing import List, Tuple  # Still needed for python before 3.9.
import numpy as np


class ResizableAutoencoder:
    """Holds the layers for an Autoencoder setup that can be specialized to a specific image size."""
    def __init__(self, n_folds: int, filter_size_schedule: List[int], n_channels: int = 3, ksize: int = 3):
        self.n_folds = n_folds
        self.n_channels = n_channels
        self.inner_model = k.Sequential()
        self.layers = []
        filter_size_index = 0
        kernel_size = (ksize, ksize)

        # Build the layers that will be later reused when building a model for a particular
        # image size.
        for i in range(self.n_folds):
            self.layers.append(k.layers.Conv2D(filters=filter_size_schedule[filter_size_index],
                                               kernel_size=kernel_size, activation=k.activations.relu))
            self.layers.append(k.layers.MaxPool2D())
            filter_size_index += 1

        for i in range(self.n_folds):
            self.layers.append(k.layers.Conv2D(filters=filter_size_schedule[filter_size_index],
                                               kernel_size=kernel_size, activation=k.activations.relu))
            self.layers.append(k.layers.UpSampling2D())
            filter_size_index += 1

        self.layers.append(
            k.layers.Conv2D(filters=n_channels, kernel_size=kernel_size, activation=k.activations.sigmoid))
        # print(f'{self.layers=}')

    def make_subimage_model(self, inner_size: int) -> k.Model:
        """Make a model for a small subimage. Mostly used for training."""
        subimage_size = self.subimage_size_from_inner_size(inner_size)
        model = k.Sequential()
        model.add(k.layers.InputLayer(input_shape=(subimage_size, subimage_size, self.n_channels)))
        for layer in self.layers:
            model.add(layer)
        return model

    def make_full_image_model(self, label_shape: Tuple[int]) -> Tuple[k.Model, Tuple[int]]:
        """Make a model for a full image. Mostly used for prediction."""
        n_inner_y = self.inner_size_from_label_size(label_shape[0])
        n_inner_x = self.inner_size_from_label_size(label_shape[1])
        image_shape = (self.subimage_size_from_inner_size(n_inner_y),
                       self.subimage_size_from_inner_size(n_inner_x),
                       self.n_channels)
        model = k.Sequential()
        model.add(k.layers.InputLayer(input_shape=image_shape))
        for layer in self.layers:
            model.add(layer)
        # The output size of the model is given by the size of the innermost layer * 2^n_folds. This may
        # fail to match a specific target size. Center crop the result to make it match.
        model.add(k.layers.experimental.preprocessing.CenterCrop(label_shape[0], label_shape[1]))

        return model, image_shape

    def load_from_model(self, saved_model):
        """Load weights and biases from a saved model.

        The model must be saved from a model produced by ResizableAutoencoder with the same settings.
        """
        # Make a model so that the layers are properly initialized.
        discarded_model = self.make_subimage_model(inner_size=8)
        discarded_model.summary()

        layer_offset = 0
        if saved_model.layers[0] is k.layers.InputLayer:
            layer_offset = 1

        for layer_index in range(len(self.layers)):
            self.layers[layer_index].set_weights(
                saved_model.layers[layer_index + layer_offset].get_weights())

    def inner_size_from_label_size(self, label_size: int) -> int:
        """Compute the image size of the innermost smallest layer given the label (output) size.

        Note that due to rounding, this is not a strict inverse of 'label_size_from_inner_size'.
        """
        return 4 + math.ceil((label_size - 2) / (2 ** self.n_folds))

    def label_size_from_inner_size(self, inner_size: int) -> int:
        """Compute the label (output) size given the size of the innermost smallest layer."""
        return (2 ** self.n_folds) * (inner_size - 4) + 2

    def subimage_size_from_inner_size(self, inner_size:int) -> int:
        """Compute the image (input) size given the size of the innermost smallest layer."""
        return (2 ** self.n_folds) * (inner_size + 2) - 2


class ResizableAutoencoderTest(unittest.TestCase):
    def setUp(self):
        self.resizable_autoencoder = \
            ResizableAutoencoder(n_folds=2, filter_size_schedule=[8, 16, 16, 8])

    def test_inner_size_from_label_size(self):
        # Test known inner size (computed on paper).
        self.assertEqual(12, self.resizable_autoencoder.inner_size_from_label_size(label_size=34))
        # Due to rounding, inner size will match for some different label sizes.
        self.assertEqual(12, self.resizable_autoencoder.inner_size_from_label_size(label_size=32))

    def test_label_size_from_inner_size(self):
        # Inverse operation of the above 'test_inner_size_from_label_size'
        self.assertEqual(34, self.resizable_autoencoder.label_size_from_inner_size(inner_size=12))

    def test_subimage_size_from_inner_size(self):
        # Test known subimage size (computed on paper).
        self.assertEqual(54, self.resizable_autoencoder.subimage_size_from_inner_size(inner_size=12))

    def test_full_image_model_size(self):
        # TODO: test other n_fold values, as they will have rounding issues too.

        def test_target_shape(target_shape):
            model, image_shape = self.resizable_autoencoder.make_full_image_model(label_shape=target_shape)
            input = np.zeros(shape=image_shape, dtype=np.float)
            result = model.predict(input[np.newaxis, ...])
            result = result[0]
            self.assertEqual(result.shape, target_shape)

        for i in range(250, 300):
            test_target_shape((i, 320, 3))

        for i in range(250, 300):
            test_target_shape((256, i, 3))

if __name__ == "__main__":
    unittest.main()