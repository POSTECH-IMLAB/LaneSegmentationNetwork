# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for creating TFRecord data sets.
source: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""

import json
import tensorflow as tf
from multiprocessing.pool import ThreadPool
from threading import Lock
from utils.preprocessing import decode_shi_labels
import numpy as np
from time import time, sleep
from PIL import Image
import sys
import os
import cv2
from utils.base_util import is_image, random_select, random_integers, Timer, file_list
from matplotlib import pyplot as plt
from random import shuffle


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_examples_list(path):
    """Read list of training or validation examples.

    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.

    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

    Args:
      path: absolute path to examples list file.

    Returns:
      list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def make_initializable_iterator(dataset):
    """Creates an iterator, and initializes tables.

    This is useful in cases where make_one_shot_iterator wouldn't work because
    the graph contains a hash table that needs to be initialized.

    Args:
      dataset: A `tf.data.Dataset` object.

    Returns:
      A `tf.data.Iterator`.
    """
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    return iterator


def read_dataset(
        file_read_func, decode_func, input_files, config, num_workers=1,
        worker_index=0):
    """Reads a dataset, and handles repetition and shuffling.

    Args:
      file_read_func: Function to use in tf.data.Dataset.interleave, to read
        every individual file into a tf.data.Dataset.
      decode_func: Function to apply to all records.
      input_files: A list of file paths to read.
      config: A input_reader_builder.InputReader object.
      num_workers: Number of workers / shards.
      worker_index: Id for the current worker.

    Returns:
      A tf.data.Dataset based on config.
    """
    # Shard, shuffle, and read files.
    filenames = tf.concat([tf.matching_files(pattern) for pattern in input_files],
                          0)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shard(num_workers, worker_index)
    dataset = dataset.repeat(config.num_epochs or None)
    if config.shuffle:
        dataset = dataset.shuffle(config.filenames_shuffle_buffer_size,
                                  reshuffle_each_iteration=True)

    # Read file records and shuffle them.
    # If cycle_length is larger than the number of files, more than one reader
    # will be assigned to the same file, leading to repetition.
    cycle_length = tf.cast(
        tf.minimum(config.num_readers, tf.size(filenames)), tf.int64)
    # TODO: find the optimal block_length.
    dataset = dataset.interleave(
        file_read_func, cycle_length=cycle_length, block_length=1)

    if config.shuffle:
        dataset = dataset.shuffle(config.shuffle_buffer_size,
                                  reshuffle_each_iteration=True)

    dataset = dataset.map(decode_func, num_parallel_calls=config.num_readers)
    return dataset.prefetch(config.prefetch_buffer_size)


def run_thread_blocks(number_of_threads, data_list, thread_fn, writer=None, timer=None, return_result=False):
    pool = ThreadPool(processes=number_of_threads)
    data_block_list = list()
    partition_size = int(len(data_list) / number_of_threads)
    for i in range(number_of_threads):
        if i != (number_of_threads - 1):
            data_block_list.append(data_list[i * partition_size:(i + 1) * partition_size])
        else:
            data_block_list.append(data_list[i * partition_size:])

    threads = list()
    for i in range(number_of_threads):
        threads.append(pool.apply_async(thread_fn, (i, data_block_list[i], writer, timer)))

    if return_result:
        outputs = list()
        for i in range(number_of_threads):
            outputs.append(threads[i].get())
        return outputs
    else:
        for i in range(number_of_threads):
            threads[i].get()


label_mask = [
    [0, 0, 255],  # Sea
    [0, 255, 0],  # Sky
    [255, 255, 255],  # Ship
    [255, 0, 0],  # Shore
    [255, 255, 0],  # Own ship
    [100, 100, 100],  # Crane
    [255, 0, 255],  # Etc
    [0, 0, 0]  # Ignore
]
label_mask = np.array(label_mask).astype(np.float32)


def convert_label(label_path, image_size, apply_ignore=True):
    """
    Convert RGB label image to onehot label image
    :param label_path: File path of RGB label image
    :param image_size: Size to resize result image
    :param apply_ignore: Apply ignore
    :return:
    """
    label = np.array(Image.open(label_path).resize((image_size[0], image_size[1]), Image.ANTIALIAS))[:, :, :3]
    converted_label = np.zeros(shape=[image_size[0], image_size[1]], dtype=np.float32)
    for index, mask in enumerate(label_mask):
        selected = np.all(label == mask, axis=2, keepdims=False)
        converted_label += selected.astype(np.uint8) * index

    ignored_part = np.equal(converted_label, len(label_mask) - 1).astype(np.float32)
    if apply_ignore:
        converted_label += ignored_part * (255 - len(label_mask) + 1)

    return converted_label


def convert_to_nearest_label(label_path, image_size, apply_ignore=True):
    """
    Convert RGB label image to onehot label image
    :param label_path: File path of RGB label image
    :param image_size: Size to resize result image
    :param apply_ignore: Apply ignore
    :return:
    """
    label = np.array(Image.open(label_path).resize((image_size[0], image_size[1]), Image.ANTIALIAS))[:, :, :3]
    label = label.astype(np.float32)
    stacked_label = list()
    for index, mask in enumerate(label_mask):
        length = np.sum(cv2.pow(label - mask, 2), axis=2, keepdims=False)
        length = cv2.sqrt(length)
        stacked_label.append(length)

    stacked_label = np.array(stacked_label)
    stacked_label = np.transpose(stacked_label, [1, 2, 0])
    converted_to_classes = np.argmin(stacked_label, axis=2).astype(np.uint8)
    if apply_ignore:
        ignore_mask = (converted_to_classes == (len(label_mask) - 1)).astype(np.uint8)
        ignore_mask *= (256 - len(label_mask))
        converted_to_classes += ignore_mask

    return converted_to_classes


def make_shi_tfrecord(image_path, label_path, train_data_output_path, eval_data_output_path, shuffle_data=True):
    """
    Multi thread version of tfrecord maker
    :param image_path: Path of images
    :param label_path: Path of image dataset
    :param train_data_output_path: Path of train tfrecord
    :param eval_data_output_path: Path of eval tfrecord
    :param shuffle_data: To shuffle data or not
    :return:
    """
    if isinstance(image_path, str):
        image_folder_list = [os.path.join(image_path, l) for l in os.listdir(image_path)
                             if is_image(os.path.join(image_path, l))]
    else:
        raise ValueError('Need to provide path of image set in string or list of images')

    if isinstance(label_path, str):
        label_folder_list = [os.path.join(label_path, l) for l in os.listdir(label_path)
                             if is_image(os.path.join(label_path, l))]
    else:
        raise ValueError('Need to provide path of image set in string or list of images')
    print('Make tfrecord from %s to %s' % (label_path, train_data_output_path))

    image_and_label_list = [[i, j] for [i, j] in zip(image_folder_list, label_folder_list)]
    if shuffle_data:
        shuffle(image_and_label_list)

    eval_list, train_list = random_select(image_and_label_list, 200, return_unselected=True)

    train_writer = tf.python_io.TFRecordWriter(train_data_output_path)
    eval_writer = tf.python_io.TFRecordWriter(eval_data_output_path)

    # Make list of image data
    print('Start make image and label pairs')

    mutex = Lock()
    mutex2 = Lock()

    image_size = (2048, 2048)
    division = 1

    def load_image_pairs(thread_id, img_and_label_list, writer=None, timer=None):
        global finished_samples
        global time_mean
        global time_start
        if thread_id == 0:
            time_start = time()
            finished_samples = 0
            time_mean = -1
        _height = int(image_size[0] / division)
        _width = int(image_size[1] / division)
        for idx, (img, lab) in enumerate(img_and_label_list):
            image = np.array(Image.open(img).convert('RGB').resize((image_size[0], image_size[1]), Image.ANTIALIAS))
            converted_label = convert_to_nearest_label(lab, image_size)

            divided_images = list()
            divided_labels = list()
            for d_i in range(division * 2 - 1):
                for d_j in range(division * 2 - 1):
                    selected_label = converted_label[
                        d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                        d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width
                    ]

                    if np.sum(np.equal(selected_label, 2).astype(np.int32)) == 0:
                        continue

                    divided_images.append(
                        np.array(
                            image[
                                d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                                d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width,
                                :3
                            ],
                            dtype=np.uint8
                        )
                    )

                    divided_labels.append(
                        np.array(selected_label, dtype=np.uint8)
                    )

                    if np.shape(divided_images[-1]) != (image_size[0], image_size[1], 3) or \
                        np.shape(divided_labels[-1]) != (image_size[0], image_size[1]):
                        print(np.shape(divided_images[-1]), np.shape(divided_labels[-1]))
                        raise ValueError('Shape of image or label are inappropriate (%s, %s)' % (img, lab))

            if len(divided_images) != len(divided_labels):
                raise ValueError('Number of divided images and labels are different! (%s, %s)' % (img, lab))

            if len(divided_images) > 0:
                for d_i, d_l in zip(divided_images, divided_labels):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height': int64_feature(_height),
                                'width': int64_feature(_width),
                                'image/raw': bytes_feature(d_i.tostring()),
                                'label/raw': bytes_feature(d_l.tostring())
                            }
                        )
                    )
                    if writer is not None:
                        mutex.acquire()
                        writer.write(example.SerializeToString())  # Write to tfrecord
                        mutex.release()

            mutex2.acquire()
            if timer is not None:
                timer.tick_timer()
            mutex2.release()

    timer = Timer()

    def print_function():
        sys.stdout.write('\rLoad images : [%d/%d][%.2f%%][%d hour %d minute %d second left]' %
                         (timer.current_progress, timer.whole_number_of_data,
                          timer.current_progress / timer.whole_number_of_data * 100,
                          int(timer.remain_time / 3600), int(timer.remain_time / 60) % 60, timer.remain_time % 60))
        sys.stdout.flush()

    timer.print_fn = print_function

    timer.start(len(train_list))
    run_thread_blocks(
        number_of_threads=16,
        data_list=train_list,
        thread_fn=load_image_pairs,
        writer=train_writer,
        timer=timer
    )

    timer.start(len(eval_list))
    run_thread_blocks(
        number_of_threads=16,
        data_list=eval_list,
        thread_fn=load_image_pairs,
        writer=eval_writer,
        timer=timer
    )

    print('\nCreate tfrecord [%s] finished!' % train_data_output_path)
    print('Create tfrecord [%s] finished!' % eval_data_output_path)
    train_writer.close()
    eval_writer.close()


def make_lane_tfrecord(data_path, data_output_path, shuffle_data=True):
    """
    Multi thread version of tfrecord maker
    :param data_path: Path of data
    :param data_output_path: Path of tfrecord
    :param shuffle_data: To shuffle data or not
    :return:
    """

    data_folders = file_list(data_path)

    image_list = list()
    label_list = list()
    for f in data_folders:
        _folder = file_list(f)
        image_list += file_list(_folder[1])
        label_list += file_list(_folder[0])

    image_and_label_list = [[i, j] for [i, j] in zip(image_list, label_list)]
    if shuffle_data:
        shuffle(image_and_label_list)

    print('Make tfrecord from %s to %s' % (data_path, data_output_path))
    data_writer = tf.python_io.TFRecordWriter(data_output_path)

    # Make list of image data
    print('Start make image and label pairs')

    mutex = Lock()
    mutex2 = Lock()

    image_size = (720, 1080)
    division = 1

    def load_image_pairs(thread_id, img_and_label_list, writer=None, timer=None):
        global finished_samples
        global time_mean
        global time_start
        if thread_id == 0:
            time_start = time()
            finished_samples = 0
            time_mean = -1
        _height = int(image_size[0] / division)
        _width = int(image_size[1] / division)
        for idx, (img, lab) in enumerate(img_and_label_list):
            image = np.array(Image.open(img).convert('RGB'))
            label = np.array(Image.open(lab))
            label = np.equal(label, 2).astype(np.uint8) + np.equal(label, 255).astype(np.uint8) * 255

            divided_images = list()
            divided_labels = list()
            for d_i in range(division * 2 - 1):
                for d_j in range(division * 2 - 1):
                    divided_images.append(
                        np.array(
                            image[
                                d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                                d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width,
                                :3
                            ],
                            dtype=np.uint8
                        )
                    )

                    divided_labels.append(
                        np.array(
                            label[
                                d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                                d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width
                            ],
                            dtype=np.uint8
                        )
                    )

                    if np.shape(divided_images[-1]) != (image_size[0], image_size[1], 3) or \
                        np.shape(divided_labels[-1]) != (image_size[0], image_size[1]):
                        print(np.shape(divided_images[-1]), np.shape(divided_labels[-1]))
                        raise ValueError('Shape of image or label are inappropriate (%s, %s)' % (img, lab))

            if len(divided_images) != len(divided_labels):
                raise ValueError('Number of divided images and labels are different! (%s, %s)' % (img, lab))

            if len(divided_images) > 0:
                for d_i, d_l in zip(divided_images, divided_labels):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height': int64_feature(_height),
                                'width': int64_feature(_width),
                                'image/raw': bytes_feature(d_i.tostring()),
                                'label/raw': bytes_feature(d_l.tostring())
                            }
                        )
                    )
                    if writer is not None:
                        mutex.acquire()
                        writer.write(example.SerializeToString())  # Write to tfrecord
                        mutex.release()

            mutex2.acquire()
            if timer is not None:
                timer.tick_timer()
            mutex2.release()

    timer = Timer()

    def print_function():
        sys.stdout.write('\rLoad images : [%d/%d][%.2f%%][%d hour %d minute %d second left]' %
                         (timer.current_progress, timer.whole_number_of_data,
                          timer.current_progress / timer.whole_number_of_data * 100,
                          int(timer.remain_time / 3600), int(timer.remain_time / 60) % 60, timer.remain_time % 60))
        sys.stdout.flush()

    timer.print_fn = print_function

    timer.start(len(image_and_label_list))

    run_thread_blocks(
        number_of_threads=16,
        data_list=image_and_label_list,
        thread_fn=load_image_pairs,
        writer=data_writer,
        timer=timer
    )

    print('\nCreate tfrecord [%s] finished!' % data_output_path)
    data_writer.close()


def make_culane_tfrecord(data_path, data_output_path, shuffle_data=True):
    """
    Multi thread version of tfrecord maker
    :param data_path: Path of data
    :param data_output_path: Path of tfrecord
    :param shuffle_data: To shuffle data or not
    :return:
    """

    data_folders = file_list(data_path)
    image_folders = file_list(data_folders[0])
    label_folders = file_list(data_folders[1])

    image_list = list()
    label_list = list()
    for i_f, l_f in zip(image_folders, label_folders):
        image_list += file_list(i_f)
        label_list += file_list(l_f)

    image_and_label_list = [[i, j] for [i, j] in zip(image_list, label_list)]
    image_and_label_list = random_select(image_and_label_list, size=30000)
    if shuffle_data:
        shuffle(image_and_label_list)

    print('Make tfrecord from %s to %s' % (data_path, data_output_path))
    data_writer = tf.python_io.TFRecordWriter(data_output_path)

    # Make list of image data
    print('Start make image and label pairs')

    mutex = Lock()
    mutex2 = Lock()

    image_size = (590, 1640)
    division = 1

    def load_image_pairs(thread_id, img_and_label_list, writer=None, timer=None):
        global finished_samples
        global time_mean
        global time_start
        if thread_id == 0:
            time_start = time()
            finished_samples = 0
            time_mean = -1
        _height = int(image_size[0] / division)
        _width = int(image_size[1] / division)
        for idx, (img, lab) in enumerate(img_and_label_list):
            image = np.array(Image.open(img).convert('RGB'))
            label = np.array(Image.open(lab))
            label = np.not_equal(label, 0).astype(np.uint8)

            divided_images = list()
            divided_labels = list()
            for d_i in range(division * 2 - 1):
                for d_j in range(division * 2 - 1):
                    divided_images.append(
                        np.array(
                            image[
                                d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                                d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width,
                                :3
                            ],
                            dtype=np.uint8
                        )
                    )

                    divided_labels.append(
                        np.array(
                            label[
                                d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                                d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width
                            ],
                            dtype=np.uint8
                        )
                    )

                    if np.shape(divided_images[-1]) != (image_size[0], image_size[1], 3) or \
                        np.shape(divided_labels[-1]) != (image_size[0], image_size[1]):
                        print(np.shape(divided_images[-1]), np.shape(divided_labels[-1]))
                        raise ValueError('Shape of image or label are inappropriate (%s, %s)' % (img, lab))

            if len(divided_images) != len(divided_labels):
                raise ValueError('Number of divided images and labels are different! (%s, %s)' % (img, lab))

            if len(divided_images) > 0:
                for d_i, d_l in zip(divided_images, divided_labels):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height': int64_feature(_height),
                                'width': int64_feature(_width),
                                'image/raw': bytes_feature(d_i.tostring()),
                                'label/raw': bytes_feature(d_l.tostring())
                            }
                        )
                    )
                    if writer is not None:
                        mutex.acquire()
                        writer.write(example.SerializeToString())  # Write to tfrecord
                        mutex.release()

            mutex2.acquire()
            if timer is not None:
                timer.tick_timer()
            mutex2.release()

    timer = Timer()

    def print_function():
        sys.stdout.write('\rLoad images : [%d/%d][%.2f%%][%d hour %d minute %d second left]' %
                         (timer.current_progress, timer.whole_number_of_data,
                          timer.current_progress / timer.whole_number_of_data * 100,
                          int(timer.remain_time / 3600), int(timer.remain_time / 60) % 60, timer.remain_time % 60))
        sys.stdout.flush()

    timer.print_fn = print_function

    timer.start(len(image_and_label_list))

    run_thread_blocks(
        number_of_threads=16,
        data_list=image_and_label_list,
        thread_fn=load_image_pairs,
        writer=data_writer,
        timer=timer
    )

    print('\nCreate tfrecord [%s] finished!' % data_output_path)
    data_writer.close()


def make_bdd100k_tfrecord(data_path, label_path, data_output_path, shuffle_data=True):
    """
    Multi thread version of tfrecord maker
    :param data_path: Path of data
    :param data_output_path: Path of tfrecord
    :param shuffle_data: To shuffle data or not
    :return:
    """

    print('[Lead json]')
    data = open(label_path).read()
    print('[Done]')

    print('[Load json]')
    data = json.loads(data)
    print('[Done]')

    print('[Gather data]')
    dataset = list()
    for d in data:
        vertices = list()
        closed = list()
        for l in d['labels']:
            if l['category'] == 'lane' and (
                    l['attributes']['laneType'] == 'single white' or
                    l['attributes']['laneType'] == 'single yellow' or
                    l['attributes']['laneType'] == 'double white' or
                    l['attributes']['laneType'] == 'double yellow'
            ):
                if len(l['poly2d']) != 1:
                    raise ValueError('Poly2d!!', d['name'])
                vertices.append(l['poly2d'][0]['vertices'])
                closed.append(l['poly2d'][0]['closed'])

        if len(vertices) == 0:
            continue

        _d = {
            'image_path': os.path.join(data_path, d['name']),
            'vertices': vertices,
            'closed': closed
        }
        dataset.append(_d)
    print('[Done]')

    print('[Random select and shuffle data]')
    dataset = random_select(dataset, 30000)
    if shuffle_data:
        shuffle(dataset)
    print('[Done]')
    print('Number of data', len(dataset))

    print('Make tfrecord from %s to %s' % (data_path, data_output_path))
    data_writer = tf.python_io.TFRecordWriter(data_output_path)

    # Make list of image data
    print('Start make image and label pairs')

    mutex = Lock()
    mutex2 = Lock()

    image_size = (720, 1280)
    division = 1

    def load_image_pairs(thread_id, img_and_label_list, writer=None, timer=None):
        global finished_samples
        global time_mean
        global time_start
        if thread_id == 0:
            time_start = time()
            finished_samples = 0
            time_mean = -1
        _height = int(image_size[0] / division)
        _width = int(image_size[1] / division)
        for idx, _data in enumerate(img_and_label_list):
            image = np.array(Image.open(_data['image_path']).convert('RGB'))
            label = np.zeros_like(image, dtype=np.uint8)
            for v, c in zip(_data['vertices'], _data['closed']):
                cv2.polylines(label, np.array([v], dtype=np.int32), c, 1, thickness=12)

            divided_images = list()
            divided_labels = list()
            for d_i in range(division * 2 - 1):
                for d_j in range(division * 2 - 1):
                    divided_images.append(
                        np.array(
                            image[
                                d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                                d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width,
                                :3
                            ],
                            dtype=np.uint8
                        )
                    )

                    divided_labels.append(
                        np.array(
                            label[
                                d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                                d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width
                            ],
                            dtype=np.uint8
                        )
                    )

                    if np.shape(divided_images[-1]) != (image_size[0], image_size[1], 3) or \
                        np.shape(divided_labels[-1]) != (image_size[0], image_size[1]):
                        print(np.shape(divided_images[-1]), np.shape(divided_labels[-1]))
                        raise ValueError('Shape of image or label are inappropriate (%s)' % (_data['image_path']))

            if len(divided_images) != len(divided_labels):
                raise ValueError('Number of divided images and labels are different! (%s)' % (_data['image_path']))

            if len(divided_images) > 0:
                for d_i, d_l in zip(divided_images, divided_labels):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height': int64_feature(_height),
                                'width': int64_feature(_width),
                                'image/raw': bytes_feature(d_i.tostring()),
                                'label/raw': bytes_feature(d_l.tostring())
                            }
                        )
                    )
                    if writer is not None:
                        mutex.acquire()
                        writer.write(example.SerializeToString())  # Write to tfrecord
                        mutex.release()

            mutex2.acquire()
            if timer is not None:
                timer.tick_timer()
            mutex2.release()

    timer = Timer()

    def print_function():
        sys.stdout.write('\rLoad images : [%d/%d][%.2f%%][%d hour %d minute %d second left]' %
                         (timer.current_progress, timer.whole_number_of_data,
                          timer.current_progress / timer.whole_number_of_data * 100,
                          int(timer.remain_time / 3600), int(timer.remain_time / 60) % 60, timer.remain_time % 60))
        sys.stdout.flush()

    timer.print_fn = print_function

    timer.start(len(dataset))

    run_thread_blocks(
        number_of_threads=16,
        data_list=dataset,
        thread_fn=load_image_pairs,
        writer=data_writer,
        timer=timer
    )

    print('\nCreate tfrecord [%s] finished!' % data_output_path)
    data_writer.close()


def main(unused_argv):

    make_bdd100k_tfrecord(
        data_path='D:/BDD100k/bdd100k/images/100k/train',
        label_path='D:/BDD100k/bdd100k_labels_images_train.json',
        data_output_path='D:/BDD100k/lane_train.tfrecord',
        shuffle_data=True
    )

    # make_culane_tfrecord(
    #     data_path='D:/CULane',
    #     data_output_path='D:/CULane/lane_train.tfrecord',
    #     shuffle_data=True
    # )

    # make_lane_tfrecord(
    #     data_path='D:/HighwayDriving/Train',
    #     data_output_path='C:/Users/YDK/Desktop/YDK/Graduate School/Work/STAR Lab/LaneSegmentationNetwork' +
    #                      '/dataset_lane/lane_train.tfrecord',
    #     shuffle_data=True
    # )
    # make_lane_tfrecord(
    #     data_path='D:/HighwayDriving/Test',
    #     data_output_path='C:/Users/YDK/Desktop/YDK/Graduate School/Work/STAR Lab/LaneSegmentationNetwork' +
    #                      '/dataset_lane/lane_test.tfrecord',
    #     shuffle_data=True
    # )
    # make_shi_tfrecord(image_path='D:/Samsung Heavy Industries/DB/original',
    #                   label_path='D:/Samsung Heavy Industries/DB/LabelsConverted',
    #                   train_data_output_path='D:/Samsung Heavy Industries/DB/LabelsConverted/' +
    #                                          'SHI_dataset_ori.tfrecord',
    #                   eval_data_output_path='D:/Samsung Heavy Industries/DB/LabelsConverted/' +
    #                                         'SHI_dataset_eval_ori.tfrecord',
    #                   shuffle_data=True)


if __name__ == '__main__':
    tf.app.run(main=main)
