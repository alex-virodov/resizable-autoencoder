from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *
import numpy as np
import cv2
import itertools
from util.cv2util import cv2_upscale

def pad_and_set_max(list, index, value):
    while (len(list)) < index+1:
        list.append(0)
    list[index] = max(value, list[index])

def pad_and_set_max_with_span(list, index, value, span):
    if (span == 1):
        pad_and_set_max(list, index, value)
    else:
        for i in range(index, index + span):
            pad_and_set_max(list, i, 0)

def fix_ranges(canvas_start, canvas_end, image_width):
    # While we pick up the max in each dimension, canvas_width >= image_width.
    canvas_width = canvas_end - canvas_start
    # print(f"canvas_start={canvas_start} canvas_end={canvas_end} image_width={image_width} canvas_width={canvas_width}")
    if image_width == canvas_width:
        return (canvas_start, canvas_end, 0, image_width)
    else:
        return (canvas_start, canvas_start + image_width, 0, image_width)

def fix_for_span(sizes, index, span, image_size):
    canvas_size = sum(sizes[index:index + span])
    if (canvas_size < image_size):
        sizes[index + span - 1] += image_size - canvas_size


def copy_to_canvas(canvas, col_starts, row_starts, x, y, image, y_offset):
    canvas1_x = col_starts[x]
    canvas1_y = row_starts[y] + y_offset
    canvas2_x = col_starts[x + 1]
    canvas2_y = row_starts[y + 1]
    canvas[canvas1_y:canvas2_y, canvas1_x:canvas2_x] = (10,10,50)
    canvas1_x, canvas2_x, image1_x, image2_x = fix_ranges(canvas1_x, canvas2_x, image.shape[1])
    canvas1_y, canvas2_y, image1_y, image2_y = fix_ranges(canvas1_y, canvas2_y, image.shape[0])

    if len(image.shape) == 2:
        canvas[canvas1_y:canvas2_y, canvas1_x:canvas2_x, 0] = image[image1_y:image2_y, image1_x:image2_x]
        canvas[canvas1_y:canvas2_y, canvas1_x:canvas2_x, 1] = image[image1_y:image2_y, image1_x:image2_x]
        canvas[canvas1_y:canvas2_y, canvas1_x:canvas2_x, 2] = image[image1_y:image2_y, image1_x:image2_x]
    else:
        canvas[canvas1_y:canvas2_y, canvas1_x:canvas2_x] = image[image1_y:image2_y, image1_x:image2_x]

    # return rendered location
    return (canvas1_x, canvas1_y, canvas2_x, canvas2_y)


class Cv2GridCanvas:
    def __init__(self, top_margin=1):
        self.images = {}
        self.original_images = {}
        self.spans = {}
        self.titles = {}
        self.upscale = {}
        self.rendered_location = {}
        self.canvas = None
        self.top_margin = top_margin

    def imshow(self, x, y, image, title='', colspan=1, rowspan=1, upscale=1):
        self.images[(x, y)] = image if upscale == 1 else cv2_upscale(image, upscale)
        self.original_images[(x, y)] = image
        self.spans[(x,y)] = (colspan, rowspan)
        self.titles[(x,y)] = title
        self.upscale[(x, y)] = upscale
        self.rendered_location[(x, y)] = (0, 0, 0, 0)

    def render_image(self):
        col_sizes = []
        row_sizes = []

        for (x,y) in self.images:
            shape = self.images[(x,y)].shape
            spans = self.spans[(x,y)]
            pad_and_set_max_with_span(col_sizes, x, shape[1], spans[0])
            pad_and_set_max_with_span(row_sizes, y, shape[0] + self.top_margin, spans[1])

        # print(f"pre-span col_sizes={col_sizes} row_sizes={row_sizes}")

        # Another pass to extend rows/cols to accommodate spanned images
        for (x,y) in self.images:
            shape = self.images[(x,y)].shape
            spans = self.spans[(x,y)]
            fix_for_span(col_sizes, x, spans[0], shape[1])
            fix_for_span(row_sizes, y, spans[1], shape[0] + self.top_margin)

        # print(f" post-span col_sizes={col_sizes} row_sizes={row_sizes}")
        col_starts = [0] + list(itertools.accumulate(col_sizes))
        row_starts = [0] + list(itertools.accumulate(row_sizes))
        # print(f"col_starts={col_starts} row_starts={row_starts}")

        width = sum(col_sizes)
        height = sum(row_sizes)

        if self.canvas is None \
                or self.canvas.shape[0] != height \
                or self.canvas.shape[1] != width \
                or self.canvas.shape[0] != 3:
            self.canvas = np.zeros((height, width, 3), np.uint8)

        for (x,y) in self.images:
            self.rendered_location[(x, y)] = copy_to_canvas(
                self.canvas, col_starts, row_starts, x, y, self.images[(x,y)], y_offset=self.top_margin)
            title = self.titles[(x,y)]
            if len(title) > 0:
                # cv2.putText(self.canvas, self.titles[(x,y)] + " (" + str(self.images[(x,y)].dtype) + ")", (col_starts[x], row_starts[y] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                cv2.putText(self.canvas, self.titles[(x, y)],
                            (col_starts[x], row_starts[y] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if False:
            for col_start in col_starts:
                cv2.line(self.canvas, (col_start, 0), (col_start, height), (255,255,0))

            for row_start in row_starts:
                cv2.line(self.canvas, (0, row_start), (width, row_start), (255,255,0))
        return self.canvas

    def render(self, window):
        cv2.imshow(window, self.render_image())

    def on_mouse(self, event, x, y, flags, param):
        # Windows-specific.
        # import win32api
        # import win32con
        # win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_UPARROW))
        if event == cv2.EVENT_LBUTTONDOWN:
            # find index (x,y) from mouse position
            for (ix,iy) in self.images:
                range_x1 = self.rendered_location[(ix, iy)][0]
                range_y1 = self.rendered_location[(ix, iy)][1]
                range_x2 = self.rendered_location[(ix, iy)][2]
                range_y2 = self.rendered_location[(ix, iy)][3]

                if x >= range_x1 and x < range_x2 and y >= range_y1 and y < range_y2:
                    x -= range_x1
                    y -= range_y1
                    x /= self.upscale[(ix, iy)]
                    y /= self.upscale[(ix, iy)]
                    value = self.original_images[(ix, iy)][int(y), int(x)]
                    print(f'on_mouse event={event} x={x} y={y} value={value} flags={flags} param={param}')



if __name__ == "__main__":
    # Example usage.
    image = np.array([[255, 0, 255], [0, 128, 0], [64, 0, 64]])
    canvas = Cv2GridCanvas(top_margin=30)
    canvas.imshow(0, 0, image, title='test', upscale=32)
    canvas.render('canvas test')
    cv2.waitKey(0)
    cv2.destroyAllWindows()