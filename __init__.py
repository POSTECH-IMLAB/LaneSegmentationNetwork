from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from network.lane_segmentator import Segmentator
from utils import preprocessing
from tensorflow.python import debug as tf_debug

import shutil


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_gpu', 2,
                     'Number of GPUs to use.')

flags.DEFINE_string('base_architecture', 'resnet_v2_50',
                    'The architecture of base Resnet building block.')

flags.DEFINE_string('pre_trained_model',
                    './init_checkpoints/' + FLAGS.base_architecture + '/' + FLAGS.base_architecture + '.ckpt',
                    'The architecture of base Resnet building block.')

flags.DEFINE_string('model_dir', './model',
                    'Base directory for the model')

flags.DEFINE_string('train_data', './dataset_lane/lane_train.tfrecord',
                    'Tensorflow record file for training.')

flags.DEFINE_string('test_data', './dataset_lane/lane_test.tfrecord',
                    'Tensorflow record file for testing')

flags.DEFINE_boolean('clean_model_dir', False,
                     'Whether to clean up the model directory if present.')

flags.DEFINE_integer('train_epochs', 50,
                     'Number of training epochs.')

flags.DEFINE_integer('epochs_per_eval', 5,
                     'The number of training epochs to run between evaluations.')

flags.DEFINE_integer('batch_size', 8,
                     'Size of batch.')

flags.DEFINE_integer('max_iter', 32000,
                     'Number of maximum iteration used for "poly" learning rate policy.')

flags.DEFINE_integer('initial_global_step', 0,
                     'Initial global step for controlling learning rate when fine-tuning model.')

flags.DEFINE_integer('output_stride', 16,
                     'Output stride for DeepLab v3. Currently 8 or 16 is supported.')

flags.DEFINE_float('initial_learning_rate', 0.007,
                   'Initial learning rate for the optimizer.')

flags.DEFINE_float('end_learning_rate', 0,
                   'End learning rate for the optimizer.')

flags.DEFINE_float('power', 0.9,
                   'Parameter for polynomial learning rate policy.')

flags.DEFINE_float('momentum', 0.9,
                   'Parameter for momentum optimizer.')

flags.DEFINE_float('weight_decay', 0.0005,
                   'The weight decay to use for regularizing the model.')

flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Whether fine tune parameters of batch normalization.')

flags.DEFINE_float('batch_norm_decay', 0.9997,
                   'Batch normalization decay rate.')

flags.DEFINE_boolean('debug', False,
                     'Whether to use debugger to track down bad values during training.')

flags.DEFINE_integer('num_classes', 2,
                     'Number of classes to predict.')

flags.DEFINE_integer('input_height', 720,
                     'Input images height.')

flags.DEFINE_integer('input_width', 1080,
                     'Input images width.')

flags.DEFINE_integer('input_depth', 3,
                     'Input images depth.')

flags.DEFINE_float('min_scale', 0.5,
                   'Minimum scale for multi scale input.')

flags.DEFINE_float('max_scale', 2.0,
                   'Maximum scale for multi scale input.')

flags.DEFINE_integer('ignore_label', 255,
                     'Maximum scale for multi scale input.')

_PROB_OF_FLIP = 0.5
_MEAN_RGB = [123.15, 115.90, 103.06]


def get_filenames(is_training):
    """Return a list of filenames.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: path to the the directory containing the input data.

    Returns:
      A list of file names.
    """
    if is_training:
        return [FLAGS.train_data]
    else:
        return [FLAGS.test_data]


def parse_record(raw_record):
    """Parse PASCAL image and label from a tf record."""
    keys_to_features = {
        'height':
            tf.FixedLenFeature((), tf.int64),
        'width':
            tf.FixedLenFeature((), tf.int64),
        'image/raw':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'label/raw':
            tf.FixedLenFeature((), tf.string, default_value='')
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    def reshape_rgb(tensor):
        return tf.reshape(tensor, [tf.cast(parsed['height'], tf.int32), tf.cast(parsed['width'], tf.int32), 3])

    def reshape_gray(tensor):
        return tf.reshape(tensor, [tf.cast(parsed['height'], tf.int32), tf.cast(parsed['width'], tf.int32), 1])

    image = tf.decode_raw(parsed['image/raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = reshape_rgb(image)

    label = tf.decode_raw(parsed['label/raw'], tf.uint8)
    label = tf.cast(label, tf.float32)
    label = reshape_gray(label)

    return image, label


def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0.,
                               ignore_label=255,
                               is_training=True,
                               model_variant=None):
    """
    Preprocesses the image and label.

    Args:
      image: Input image.
      label: Ground truth annotation label.
      crop_height: The height value used to crop the image and label.
      crop_width: The width value used to crop the image and label.
      min_resize_value: Desired size of the smaller image side.
      max_resize_value: Maximum allowed size of the larger image side.
      resize_factor: Resized dimensions are multiple of factor plus one.
      min_scale_factor: Minimum scale factor value.
      max_scale_factor: Maximum scale factor value.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      ignore_label: The label value which will be ignored for training and
        evaluation.
      is_training: If the preprocessing is used for training or not.
      model_variant: Model variant (string) for choosing how to mean-subtract the
        images. See feature_extractor.network_map for supported model variants.

    Returns:
      original_image: Original image (could be resized).
      processed_image: Preprocessed image.
      label: Preprocessed ground truth segmentation label.

    Raises:
      ValueError: Ground truth label not provided during training.
    """
    if is_training and label is None:
        raise ValueError('During training, label must be provided.')
    if model_variant is None:
        tf.logging.warning('Default mean-subtraction is performed. Please specify '
                           'a model_variant. See feature_extractor.network_map for '
                           'supported model variants.')

    # Keep reference to original image.
    original_image = image

    processed_image = tf.cast(image, tf.float32)

    if label is not None:
        label = tf.cast(label, tf.int32)

    # Resize image and label to the desired range.
    if min_resize_value is not None or max_resize_value is not None:
        [processed_image, label] = (
            preprocessing.resize_to_range(
                image=processed_image,
                label=label,
                min_size=min_resize_value,
                max_size=max_resize_value,
                factor=resize_factor,
                align_corners=True))
        # The `original_image` becomes the resized image.
        original_image = tf.identity(processed_image)

    # Data augmentation by randomly scaling the inputs.
    if is_training:
        scale = preprocessing.get_random_scale(
            min_scale_factor, max_scale_factor, scale_factor_step_size)
        processed_image, label = preprocessing.randomly_scale_image_and_label(
            processed_image, label, scale)
        processed_image.set_shape([None, None, 3])

    # Pad image and label to have dimensions >= [crop_height, crop_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    # Pad image with mean pixel value.
    mean_pixel = tf.reshape(_MEAN_RGB, [1, 1, 3])
    processed_image = preprocessing.pad_to_bounding_box(
        processed_image, 0, 0, target_height, target_width, mean_pixel)

    if label is not None:
        label = preprocessing.pad_to_bounding_box(
            label, 0, 0, target_height, target_width, ignore_label)

    # Randomly crop the image and label.
    if is_training and label is not None:
        processed_image, label = preprocessing.random_crop(
            [processed_image, label], crop_height, crop_width)

    processed_image.set_shape([crop_height, crop_width, 3])

    if label is not None:
        label.set_shape([crop_height, crop_width, 1])

    if is_training:
        # Randomly left-right flip the image and label.
        processed_image, label, _ = preprocessing.flip_dim(
            [processed_image, label], _PROB_OF_FLIP, dim=1)

    return processed_image, label


def input_fn(is_training, batch_size, num_epochs=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.

    Returns:
      A tuple of images and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=batch_size * 50)

    dataset = dataset.map(parse_record)
    dataset = dataset.map(
        lambda image, label: preprocess_image_and_label(
            image, label, FLAGS.input_height, FLAGS.input_width,
            min_scale_factor=FLAGS.min_scale,
            max_scale_factor=FLAGS.max_scale,
            scale_factor_step_size=0.25,
            is_training=is_training),
        num_parallel_calls=FLAGS.num_gpu
    )

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


def input_fn_for_eval(batch_size):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      data_dir: The directory containing the input data.
      scale: Rescale image and label by factor.
      flip: Whether flip image and label or not.

    Returns:
      A tuple of images and labels.
    """
    is_training = False
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    dataset = dataset.map(parse_record)
    dataset = dataset.map(
        lambda image, label: preprocess_image_and_label(
            image, label, FLAGS.input_height, FLAGS.input_width,
            is_training=is_training),
        num_parallel_calls=FLAGS.num_gpu
    )

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size * 2)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return tf.concat([images] * 2, axis=0), tf.concat([labels] * 2, axis=0)


def main(argv):
    # Set GPU to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Remove tensorflow basic logs 3: remove all
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    if FLAGS.clean_model_dir:
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    print(FLAGS)
    # Set up network
    segmentator = Segmentator(
        params={
            'batch_norm_decay': FLAGS.batch_norm_decay,
            'base_architecture': FLAGS.base_architecture,
            'output_stride': FLAGS.output_stride,
            'pre_trained_model': FLAGS.pre_trained_model,
            'num_classes': FLAGS.num_classes,
            'batch_size': int(FLAGS.batch_size / FLAGS.num_gpu),
            'weight_decay': FLAGS.weight_decay,
            'initial_learning_rate': FLAGS.initial_learning_rate,
            'initial_global_step': FLAGS.initial_global_step,
            'max_iter': FLAGS.max_iter,
            'end_learning_rate': FLAGS.end_learning_rate,
            'power': FLAGS.power,
            'momentum': FLAGS.momentum,
            'fine_tune_batch_norm': FLAGS.fine_tune_batch_norm,
            'ignore_label': FLAGS.ignore_label
        }
    )
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=10,
        log_step_count_steps=10,
        save_checkpoints_secs=10000
    )

    estimator = tf.estimator.Estimator(
        model_fn=segmentator.model_fn if FLAGS.num_gpu == 1 else
            tf.contrib.estimator.replicate_model_fn(segmentator.model_fn),
        config=run_config
    )

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'Summary/learning_rate',
            'cross_entropy': 'Summary/loss',
            'train_px_accuracy': 'Summary/train_px_accuracy',
            'train_mean_iou': 'Summary/train_mean_iou',
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1)
        train_hooks = [logging_hook]
        eval_hooks = None

        if FLAGS.debug:
            debug_hook = tf_debug.LocalCLIDebugHook()
            train_hooks.append(debug_hook)
            eval_hooks = [debug_hook]

        tf.logging.info("Start training.")
        estimator.train(
            input_fn=lambda: input_fn(True, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=train_hooks,
            # steps=1  # For debug
        )

        tf.logging.info("Start evaluation.")
        eval_results = estimator.evaluate(
            input_fn=lambda: input_fn_for_eval(FLAGS.batch_size),
            hooks=eval_hooks,
            # steps=1  # For debug
        )
        print(eval_results)

    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    for s in scales:
        eval_results = estimator.evaluate(
            input_fn=lambda: input_fn_for_eval(FLAGS.batch_size * 2),
            # steps=1  # For debug
        )
        print('Scale:', s, ':', eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
