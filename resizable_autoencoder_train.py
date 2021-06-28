from resizable_autoencoder import ResizableAutoencoder
from resizable_autoencoder_model import \
    make_resizable_autoencoder, weight_edges_binary_crossentropy, load_resizable_autoencoder
import numpy as np
import random
import tensorflow.keras as k
import cv2
from util.image_cache import image_cache
from util.pad_image import pad_image
from util.cv2canvas import Cv2GridCanvas
from util.cv2util import cv2_waitKey
from typing import Tuple


def extract_random_subimages(image: np.ndarray, subimage_size: int, label: np.ndarray, label_size: int, n: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    subimages = np.zeros(shape=(n, subimage_size, subimage_size, image.shape[2]), dtype=image.dtype)
    sublabels = np.zeros(shape=(n, label_size, label_size, label.shape[2]), dtype=label.dtype)
    for i in range(n):
        half_w = subimage_size // 2
        x = random.randint(half_w, image.shape[1] - half_w)
        y = random.randint(half_w, image.shape[0] - half_w)
        subimages[i, ...] = image[y-half_w:y+half_w, x-half_w:x+half_w, :]
        half_w = label_size // 2
        sublabels[i, ...] = label[y-half_w:y+half_w, x-half_w:x+half_w, :]

    return subimages, sublabels


def train() -> ResizableAutoencoder:
    resizable_autoencoder = make_resizable_autoencoder()

    inner_size = 32
    subimage_size = resizable_autoencoder.subimage_size_from_inner_size(inner_size)
    label_size = resizable_autoencoder.label_size_from_inner_size(inner_size)
    subimage_model = resizable_autoencoder.make_subimage_model(inner_size)
    subimage_model.summary()
    print(f'{subimage_size=} {label_size=}')

    image = image_cache.get_data_image(2)
    gt_label = image_cache.get_gt_label_image(2)
    # TODO: expand image first so that we get a full range on the actual label image. Currently
    #   the edges of the label image are unused. (also we should "learn" the expansion'd data).
    #   This also limits the number of folds we can train, as the input of the net is larger than the output.
    image = image / 255.0
    gt_label = gt_label / 255.0
    # TODO: Grid scanning in addition / instead of random subimages.
    # TODO: Train on multiple images.
    np.random.seed(42)
    subimages, sublabels = extract_random_subimages(image, subimage_size, gt_label, label_size, n=256)

    subimage_model.compile(optimizer=k.optimizers.Adam(learning_rate=0.001),
                           loss=weight_edges_binary_crossentropy,
                           metrics=['accuracy'])
    subimage_model.fit(subimages, sublabels, epochs=1000)
    subimage_model.save('resizable_autoencoder.h5')

    return resizable_autoencoder


def eval(resizable_autoencoder: ResizableAutoencoder) -> None:
    image = image_cache.get_data_image(2)
    gt_label = image_cache.get_gt_label_image(2)
    image = image / 255.0
    gt_label = gt_label / 255.0

    inner_size = 32
    subimage_size = resizable_autoencoder.subimage_size_from_inner_size(inner_size)
    label_size = resizable_autoencoder.label_size_from_inner_size(inner_size)
    np.random.seed(42)
    subimages, sublabels = extract_random_subimages(image, subimage_size, gt_label, label_size, n=256)
    subimage_model = resizable_autoencoder.make_subimage_model(inner_size)
    subresult = subimage_model.predict(subimages)

    full_model, image_shape = resizable_autoencoder.make_full_image_model(label_shape=gt_label.shape)
    full_model.summary()
    print(f'{image_shape=} {gt_label.shape=}')

    expanded_image = pad_image(image, shape=image_shape)
    result = full_model.predict(expanded_image[np.newaxis,...])

    # Visualize the results.
    # TODO: Implement as callback to be able to observe training process - fun :)
    # TODO: Tensorboard integration.
    canvas = Cv2GridCanvas(top_margin=30)
    canvas.imshow(0, 0, image * 255, title='image', colspan=3)
    canvas.imshow(3, 0, gt_label * 255, title='gt_label', colspan=3)
    canvas.imshow(6, 0, expanded_image * 255, title='expanded', colspan=3)
    canvas.imshow(9, 0, result[0,...] * 255, title='result', colspan=3)
    for i in range(16):
        canvas.imshow((i%4)*3 + 0, i//4+1, subimages[i, ...] * 255, title=f'sub {i}')
        canvas.imshow((i%4)*3 + 1, i//4+1, sublabels[i, ...] * 255, title=f'lab {i}')
        canvas.imshow((i%4)*3 + 2, i//4+1, subresult[i, ...] * 255, title=f'res {i}')
    canvas.render(__file__)
    cv2_waitKey(__file__, 0)
    cv2.destroyAllWindows()


def train_eval() -> None:
    eval(train())


def load_eval() -> None:
    eval(load_resizable_autoencoder('resizable_autoencoder.h5'))


if __name__ == "__main__":
    train_eval()
    # load_eval()

