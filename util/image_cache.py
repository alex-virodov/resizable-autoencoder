import cv2
import glob
from os.path import join
import os


class ImageCache:
    """Provides convenient access to the data science bowl 2018 dataset.

    Captures the file naming conventions and image reading functionality.
    Also provides caching of both file list and the images themselves, which is convenient
    and faster in the context of python console.
    """
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.specimens = sorted(os.listdir(self.root_path))
        self.images = [None] * len(self.specimens)
        self.gt_labels = [None] * len(self.specimens)

    def get_num_images(self):
        return len(self.specimens)

    def get_data_image(self, image_index: int):
        if self.images[image_index] is None:
            specimen_path = join(self.root_path, self.specimens[image_index])
            data_image_name = glob.glob(specimen_path + "/images/*.png")[0]
            self.images[image_index] = cv2.imread(data_image_name)
        return self.images[image_index]

    def get_gt_label_image(self, image_index: int):
        if self.gt_labels[image_index] is None:
            specimen_path = join(self.root_path, self.specimens[image_index])
            label_image_name = join(specimen_path, 'gt_interior_edge_background.png')
            self.gt_labels[image_index] = cv2.imread(label_image_name)
        return self.gt_labels[image_index]

    def get_image_path(self, image_index: int):
        return join(self.root_path, self.specimens[image_index])


global image_cache
image_cache = ImageCache(
    root_path='E:\\hpa-single-cell-image-classification\\data-science-bowl-2018\\stage1_train')

