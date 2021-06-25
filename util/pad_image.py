import numpy as np
import cv2
from typing import Tuple


def center_coord(image_size: int, subimage_size: int) -> Tuple[int, int]:
    """Center a subimage in image along some axis, given image and subimage sizes."""
    x0 = (image_size - subimage_size) // 2
    x1 = x0 + subimage_size
    return x0, x1


def pad_image(image: np.ndarray, shape: Tuple[int, ...], extend_pixels: bool = True):
    """Pad an image to a (larger) shape by centering the image and repeating the boundary pixels."""
    # TODO: support down-pad? For now only padding to larger image.
    # TODO: Support arbitrary tensors - extend two dimensions, preserving rest.
    # TODO: actually use extend_pixels and implement extend_pixels=False case.
    result = np.zeros(shape=shape, dtype=image.dtype)
    y0, y1 = center_coord(shape[0], image.shape[0])
    x0, x1 = center_coord(shape[1], image.shape[1])
    channels = slice(0, image.shape[2])
    result[y0:y1, x0:x1, channels] = image

    result[y0:y1, 0:x0, channels] = image[:, 0, channels][:, np.newaxis, channels]
    result[y0:y1, x1:, channels] = image[:, -1, channels][:, np.newaxis, channels]
    result[0:y0, x0:x1, channels] = image[0, :, channels][np.newaxis, :, channels]
    result[y1:, x0:x1, channels] = image[-1, :, channels][np.newaxis, :, channels]

    result[0:y0, 0:x0, channels] = image[0, 0, channels][np.newaxis, np.newaxis, channels]
    result[0:y0, x1:, channels] = image[0, -1, channels][np.newaxis, np.newaxis, channels]
    result[y1:, 0:x0, channels] = image[-1, 0, channels][np.newaxis, np.newaxis, channels]
    result[y1:, x1:, channels] = image[-1, -1, channels][np.newaxis, np.newaxis, channels]

    return result


if __name__ == "__main__":
    # Quick visual test of the implementation above.
    from util.cv2canvas import Cv2GridCanvas
    from util.cv2util import cv2_waitKey
    from util.image_cache import image_cache

    def expand_shape_of(image):
        return image.shape[0] + 32, image.shape[1] + 64, image.shape[2]

    data_image = image_cache.get_data_image(2)
    gt_labels = image_cache.get_gt_label_image(2)

    padded_data_image = pad_image(data_image, shape=expand_shape_of(data_image))
    padded_gt_labels = pad_image(gt_labels, shape=expand_shape_of(gt_labels))

    canvas = Cv2GridCanvas(top_margin=30)
    canvas.imshow(0, 0, data_image, title='image')
    canvas.imshow(1, 0, gt_labels, title='gt_labels')
    canvas.imshow(0, 1, padded_data_image, title='padded image')
    canvas.imshow(1, 1, padded_gt_labels, title='padded gt_labels')
    canvas.render(__file__)
    cv2_waitKey(__file__, 0)
    cv2.destroyAllWindows()

