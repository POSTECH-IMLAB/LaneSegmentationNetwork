from matplotlib import pyplot as plt
from scipy.misc import imresize
from os import listdir
import os
import math
import numpy as np
import cv2
import random
from time import time

INF = 9999999999999
f_zero = 1e-10


def file_list(folder_path, sort=True):
    if not os.path.isdir(folder_path):
        raise ValueError('Need to feed path of folder.')

    _list = os.listdir(folder_path)
    _list = [os.path.join(folder_path, l) for l in _list]
    if sort:
        _list.sort()

    return _list


class Timer:

    def __init__(self, as_progress_notifier=True):
        """
        If set as_progress_notifier = True, then it will be use to check progress of some processes.
        If not it will be use to simple timer.
        :param as_progress_notifier:
        """
        self.whole_number_of_data = 0
        self.current_progress = 0
        self.time_moving_average = 0
        self.elapsed_time = 0
        self.remain_time = 0
        self.tick_start_time = 0
        self.is_progress_notifier = as_progress_notifier
        self.timer_ready = False
        self.print_fn = None

    def start(self, number_of_data=None):
        if self.is_progress_notifier:
            if number_of_data is None:
                raise ValueError('You should feed number_of_data.')
            self.whole_number_of_data = number_of_data
            self.current_progress = 0
            self.timer_ready = True
            self.tick_start_time = time()
        else:
            self.tick_start_time = time()
            self.timer_ready = True

    def tick_timer(self):
        if not self.timer_ready:
            raise AttributeError('Need to initialize timer by init_timer().')
        if not self.is_progress_notifier:
            raise AttributeError('You should set as_progress_notifier to True if you want to use tick_timer().')

        if self.current_progress == 0:
            self.time_moving_average = time() - self.tick_start_time
        else:
            tick = time() - self.tick_start_time
            self.elapsed_time = self.current_progress * self.time_moving_average
            self.time_moving_average = (self.elapsed_time + tick) / (self.current_progress + 1)

        self.current_progress += 1
        self.remain_time = (self.whole_number_of_data - self.current_progress) * self.time_moving_average

        if self.print_fn is not None:
            self.print_fn()

        self.tick_start_time = time()

    def check(self, start_timer=True):
        if self.is_progress_notifier:
            raise AttributeError('You should set as_progress_notifier to False if you want to use check().')
        self.elapsed_time = time() - self.tick_start_time

        if self.print_fn is not None:
            self.print_fn()

        if start_timer:
            self.start(self.whole_number_of_data)


def random_integers(min, max, size, exclude: list=None):
    if max < min:
        raise ValueError('Max must be larger than min')
    if exclude is not None:
        if (max - min) >= size + len(exclude):
            rand_list = list(random.sample(range(min, max + 1), size + len(exclude)))
        else:
            rand_list = range(min, max + 1)
        excluded_rand_list = [r for r in rand_list if r not in exclude]
        choices = np.random.choice(excluded_rand_list, size, replace=True if len(excluded_rand_list) < size else False)
    else:
        choices = list(random.sample(range(min, max + 1), size))

    return choices


def random_select(data_list: list, size, exclude: list=None, return_unselected=False):
    selection = random_integers(0, len(data_list) - 1, size, exclude)
    selected = [data_list[i] for i in selection]
    unselected = list()
    for idx, d in enumerate(data_list):
        if idx not in selection:
            unselected.append(d)

    if return_unselected:
        return selected, unselected
    else:
        return selected


def is_image(file_name):
    file_extension = file_name.split('.')[-1]
    if file_extension in ('jpg', 'JPG', 'jpeg', 'JPEG', 'tif', 'TIF', 'png', 'PNG', 'bmp', 'BMP'):
        return True
    else:
        return False


def resize_images(path, size):
    files = listdir(path)
    n_files = len(files)
    for idx, f in enumerate(files):
        if 1500 <= idx < 2000:
            if is_image(f):
                print('[%d/%d]' % (idx + 1, n_files))
                img = imresize(
                    plt.imread(path + '/' + f),
                    size
                )
                plt.imsave(
                    path + '/' + f,
                    img
                )


def resize_image(file, size):
    if is_image(file):
        img = imresize(
            plt.imread(file),
            size
        )
        plt.imsave(
            file,
            img
        )


def extract_roi(image, img_size: tuple, roi_size: tuple):
    """
    Resize image by img_size and crop Region Of Interest
    :param image: Input image
    :param img_size: Size for resizing
    :param roi_size: Size of ROI
    :return: ROI image
    """
    image = imresize(image, img_size)
    image = image[
          int(img_size[0] * 0.7765):int(img_size[0] * 0.8365),
          int(img_size[1] * 0.235):int(img_size[1] * 0.765),
          :3
          ]
    # image = image[
    #       int(img_size[0] * 0.7365):int(img_size[0] * 0.8365),
    #       int(img_size[1] * 0.235):int(img_size[1] * 0.765),
    #       :3
    #       ]
    image = imresize(image, roi_size)
    return image


def remove_remains(img, interest_point):
    """
    Remove remains which are not adjacent with interest_point
    :param img: Input image
    :param interest_point: Center point where we want to remain
    :return: Image which adjacent with interest_point
    """
    img = img.astype(np.uint8)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    img_inv = img.copy()
    cv2.floodFill(img_inv, mask, tuple(interest_point), 0)
    img -= img_inv

    return img


def fill_hole_and_remove_remains(img, interest_point):
    """
    If interest point is hole, then fill hole and remove remains of image
    :param img: Input image
    :param interest_point: Center point where we want to remain
    :return: Image which adjacent with interest_point
    """
    _img = remove_remains(
        img=img,
        interest_point=interest_point
    )
    img = img + _img
    img[img > 0] = 1
    img = remove_remains(
        img=img,
        interest_point=interest_point
    )

    return img


def slope(point1, point2):
    point1 = f_cut(point1)
    point2 = f_cut(point2)
    if point1[0] == point2[0]:
        return INF
    return (point1[1] - point2[1]) / (point1[0] - point2[0])


def img_center(img, bbox=None, show=False):  # Bbox have left top and right bottom position
    """
    Find center of image pixels which is not 0 and bounding box
    :param img: Input image (1 channel image)
    :param bbox: Bounding box for image
    :param show: Show result of function
    :return: Center of image element and bounding box
    """
    img_height, img_width = img.shape[:2]
    sum = np.array([0, 0])
    num = 0
    left = img_width
    right = 0
    top = img_height
    bottom = 0

    if bbox is None:
        for y in range(img_height):
            for x in range(img_width):
                if img[y][x] > 0:
                    sum += np.array([x, y])
                    num += 1
                    if x < left:
                        left = x
                    if x > right:
                        right = x
                    if y < top:
                        top = y
                    if y > bottom:
                        bottom = y
    else:
        for y in range(bbox[0][1], bbox[1][1]):
            for x in range(bbox[0][0], bbox[1][0]):
                if img[y][x] > 0:
                    sum += np.array([x, y])
                    num += 1
                    if x < left:
                        left = x
                    if x > right:
                        right = x
                    if y < top:
                        top = y
                    if y > bottom:
                        bottom = y

    center = (np.round(sum / num)).astype(np.int32)

    if show:
        img_plane = img.astype(np.uint8)
        cv2.circle(
            img=img_plane,
            center=tuple(center),
            radius=3,
            color=2,
            thickness=1
        )
        plt.imshow(img_plane)
        plt.show()

    return center, bbox if bbox is not None else [[left, top], [right, bottom]]


def round_int(num):
    if type(num) in [list, tuple, dict]:
        return type(num)(np.round(num).astype(np.int32))
    else:
        return np.round(num).astype(np.int32)


def np_cast(ary, t):
    return ary.astype(t)


def f_cut(x):
    if type(x) is float:
        return 0 if math.fabs(x) <= f_zero else x
    elif type(x) is list:
        x = np.array(x)
        mask = np.logical_not(np.fabs(x) <= f_zero)
        return list(x * mask)
    elif type(x) is np.ndarray:
        mask = np.logical_not(np.fabs(x) <= f_zero)
        return x * mask
    else:
        return x


def vector_length(x: np.ndarray):
    sum = 0
    for i in range(len(x)):
        sum += math.pow(x[i], 2)
    return math.sqrt(sum)


def voc_color_map():

    def bit_at(_int: int, idx):
        return (_int & (1 << idx)) != 0

    n = 256
    cmap = list(np.zeros([n, 3], dtype=np.uint8))
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | ((bit_at(c, 0)) << (7 - j))
            g = g | ((bit_at(c, 1)) << (7 - j))
            b = b | ((bit_at(c, 2)) << (7 - j))
            c = c >> 3
        cmap[i] = [r, g, b]

    return cmap


def voc_label_to_color(label):
    cmap = list(voc_color_map())
    image = np.tile(label, [3, 1, 1])
    for c in range(21):
        for i in range(3):
            image[i][image[i] == c] = cmap[c][i]

    image = np.transpose(image, [1, 2, 0]).astype(np.uint8)
    return image
