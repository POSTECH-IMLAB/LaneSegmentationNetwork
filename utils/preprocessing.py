
"""Utility functions for preprocessing data sets."""

from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

# colour map
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

shi_colors = [
    (0, 0, 255),  # Sea
    (0, 255, 0),  # Sky
    (255, 255, 255),  # Ship
    (255, 0, 0),  # Shore
    (255, 255, 0),  # Own ship
    (100, 100, 100)  # Crane
    # (255, 0, 255)  # Etc
]


def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                              % (n, num_images)
    # outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    # for i in range(num_images):
    #     img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
    #     pixels = img.load()
    #     for j_, j in enumerate(mask[i, :, :, 0]):
    #         for k_, k in enumerate(j):
    #             if k < num_classes:
    #                 pixels[k_, j_] = label_colours[k]
    #     outputs[i] = np.array(img)

    num, h, w, c = mask.shape
    outputs = np.zeros([num, h, w, 3], dtype=np.uint8)
    for n in range(num):
        for idx, m in enumerate(label_colours):
            _m = np.array(m, dtype=np.uint8)
            current_mask = (mask[n] == idx).astype(np.uint8)
            current_mask = np.tile(current_mask, [1, 1, 3])
            current_mask *= _m
            outputs[n] += current_mask

    return outputs


def decode_shi_labels(mask):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    num, h, w, c = mask.shape
    outputs = np.zeros([num, h, w, 3], dtype=np.uint8)
    for n in range(num):
        for idx, m in enumerate(shi_colors):
            _m = np.array(m, dtype=np.uint8)
            current_mask = (mask[n] == idx).astype(np.uint8)
            current_mask = np.tile(current_mask, [1, 1, 3])
            current_mask *= _m
            outputs[n] += current_mask

    return outputs


def mean_image_addition(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    """Adds the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(axis=2, values=channels)


def mean_image_subtraction(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def random_rescale_image_and_label(image, label, min_scale, max_scale):
    """Rescale an image and label with in target scale.

    Rescales an image and label within the range of target scale.

    Args:
      image: 3-D Tensor of shape `[height, width, channels]`.
      label: 3-D Tensor of shape `[height, width, 1]`.
      min_scale: Min target scale.
      max_scale: Max target scale.

    Returns:
      Cropped and/or padded image.
      If `images` was 3-D, a 3-D float Tensor of shape
      `[new_height, new_width, channels]`.
      If `labels` was 3-D, a 3-D float Tensor of shape
      `[new_height, new_width, 1]`.
    """
    if min_scale <= 0:
        raise ValueError('\'min_scale\' must be greater than 0.')
    elif max_scale <= 0:
        raise ValueError('\'max_scale\' must be greater than 0.')
    elif min_scale >= max_scale:
        raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

    shape = tf.shape(image)
    height = tf.to_float(shape[0])
    width = tf.to_float(shape[1])
    scale = tf.random_uniform(
        [], minval=min_scale, maxval=max_scale, dtype=tf.float32)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    image = tf.image.resize_images(image, [new_height, new_width],
                                   method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    # Since label classes are integers, nearest neighbor need to be used.
    label = tf.image.resize_images(label, [new_height, new_width],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

    return image, label


def rescale_image_and_label(image, label, scale):
    """Rescale an image and label with in target scale.

    Rescales an image and label within the range of target scale.

    Args:
      image: 3-D Tensor of shape `[height, width, channels]`.
      label: 3-D Tensor of shape `[height, width, 1]`.
      min_scale: Min target scale.
      max_scale: Max target scale.

    Returns:
      Cropped and/or padded image.
      If `images` was 3-D, a 3-D float Tensor of shape
      `[new_height, new_width, channels]`.
      If `labels` was 3-D, a 3-D float Tensor of shape
      `[new_height, new_width, 1]`.
    """

    if scale == 1.0:
        return image, label
    else:
        shape = tf.shape(image)
        height = tf.to_float(shape[0])
        width = tf.to_float(shape[1])
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)
        image = tf.image.resize_images(image, [new_height, new_width],
                                       method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        # Since label classes are integers, nearest neighbor need to be used.
        label = tf.image.resize_images(label, [new_height, new_width],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

    return image, label



def random_crop_or_pad_image_and_label(image, label, crop_height, crop_width, ignore_label):
    """Crops and/or pads an image to a target width and height.

    Resizes an image to a target width and height by rondomly
    cropping the image or padding it evenly with zeros.

    Args:
      image: 3-D Tensor of shape `[height, width, channels]`.
      label: 3-D Tensor of shape `[height, width, 1]`.
      crop_height: The new height.
      crop_width: The new width.
      ignore_label: Label class to be ignored.

    Returns:
      Cropped and/or padded image.
      If `images` was 3-D, a 3-D float Tensor of shape
      `[new_height, new_width, channels]`.
    """
    label = label - ignore_label  # Subtract due to 0 padding.
    label = tf.to_float(label)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_and_label = tf.concat([image, label], axis=2)
    image_and_label_pad = tf.image.pad_to_bounding_box(
        image_and_label, 0, 0,
        tf.maximum(crop_height, image_height),
        tf.maximum(crop_width, image_width))
    image_and_label_crop = tf.random_crop(
        image_and_label_pad, [crop_height, crop_width, 4])

    image_crop = image_and_label_crop[:, :, :3]
    label_crop = image_and_label_crop[:, :, 3:]
    label_crop += ignore_label
    label_crop = tf.to_int32(label_crop)

    return image_crop, label_crop


def random_flip_left_right_image_and_label(image, label):
    """Randomly flip an image and label horizontally (left to right).

    Args:
      image: A 3-D tensor of shape `[height, width, channels].`
      label: A 3-D tensor of shape `[height, width, 1].`

    Returns:
      A 3-D tensor of the same type and shape as `image`.
      A 3-D tensor of the same type and shape as `label`.
    """
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)
    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
    label = tf.cond(mirror_cond, lambda: tf.reverse(label, [1]), lambda: label)

    return image, label


def flip_left_right_image_and_label(image, label):
    """Randomly flip an image and label horizontally (left to right).

    Args:
      image: A 3-D tensor of shape `[height, width, channels].`
      label: A 3-D tensor of shape `[height, width, 1].`

    Returns:
      A 3-D tensor of the same type and shape as `image`.
      A 3-D tensor of the same type and shape as `label`.
    """
    image = tf.reverse(image, [1])
    label = tf.reverse(label, [1])

    return image, label


def eval_input_fn(image_filenames, label_filenames=None, batch_size=1):
    """An input function for evaluation and inference.

    Args:
      image_filenames: The file names for the inferred images.
      label_filenames: The file names for the grand truth labels.
      batch_size: The number of samples per batch. Need to be 1
          for the images of different sizes.

    Returns:
      A tuple of images and labels.
    """
    # Reads an image from a file, decodes it into a dense tensor
    def _parse_function(filename, is_label):
        if not is_label:
            image_filename, label_filename = filename, None
        else:
            image_filename, label_filename = filename

        image_string = tf.read_file(image_filename)
        image = tf.image.decode_image(image_string)
        image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
        image.set_shape([None, None, 3])

        image = mean_image_subtraction(image)

        if not is_label:
            return image
        else:
            label_string = tf.read_file(label_filename)
            label = tf.image.decode_image(label_string)
            label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
            label.set_shape([None, None, 1])

            return image, label

    if label_filenames is None:
        input_filenames = image_filenames
    else:
        input_filenames = (image_filenames, label_filenames)

    dataset = tf.data.Dataset.from_tensor_slices(input_filenames)
    if label_filenames is None:
        dataset = dataset.map(lambda x: _parse_function(x, False))
    else:
        dataset = dataset.map(lambda x, y: _parse_function((x, y), True))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    if label_filenames is None:
        images = iterator.get_next()
        labels = None
    else:
        images, labels = iterator.get_next()

    return images, labels


def flip_dim(tensor_list, prob=0.5, dim=1):
    """Randomly flips a dimension of the given tensor.

    The decision to randomly flip the `Tensors` is made together. In other words,
    all or none of the images pass in are flipped.

    Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used so
    that we can control for the probability as well as ensure the same decision
    is applied across the images.

    Args:
      tensor_list: A list of `Tensors` with the same number of dimensions.
      prob: The probability of a left-right flip.
      dim: The dimension to flip, 0, 1, ..

    Returns:
      outputs: A list of the possibly flipped `Tensors` as well as an indicator
      `Tensor` at the end whose value is `True` if the inputs were flipped and
      `False` otherwise.

    Raises:
      ValueError: If dim is negative or greater than the dimension of a `Tensor`.
    """
    random_value = tf.random_uniform([])

    def flip():
        flipped = []
        for tensor in tensor_list:
            if dim < 0 or dim >= len(tensor.get_shape().as_list()):
                raise ValueError('dim must represent a valid dimension.')
            flipped.append(tf.reverse_v2(tensor, [dim]))
        return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    outputs.append(is_flipped)

    return outputs


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
    """Pads the given image with the given pad_value.

    Works like tf.image.pad_to_bounding_box, except it can pad the image
    with any given arbitrary pad value and also handle images whose sizes are not
    known during graph construction.

    Args:
      image: 3-D tensor with shape [height, width, channels]
      offset_height: Number of rows of zeros to add on top.
      offset_width: Number of columns of zeros to add on the left.
      target_height: Height of output image.
      target_width: Width of output image.
      pad_value: Value to pad the image tensor with.

    Returns:
      3-D tensor of shape [target_height, target_width, channels].

    Raises:
      ValueError: If the shape of image is incompatible with the offset_* or
      target_* arguments.
    """
    image_rank = tf.rank(image)
    image_rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong image tensor rank [Expected] [Actual]',
         3, image_rank])
    with tf.control_dependencies([image_rank_assert]):
        image -= pad_value
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]
    target_width_assert = tf.Assert(
        tf.greater_equal(
            target_width, width),
        ['target_width must be >= width'])
    target_height_assert = tf.Assert(
        tf.greater_equal(target_height, height),
        ['target_height must be >= height'])
    with tf.control_dependencies([target_width_assert]):
        after_padding_width = target_width - offset_width - width
    with tf.control_dependencies([target_height_assert]):
        after_padding_height = target_height - offset_height - height
    offset_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(after_padding_width, 0),
            tf.greater_equal(after_padding_height, 0)),
        ['target size not possible with the given target offsets'])

    height_params = tf.stack([offset_height, after_padding_height])
    width_params = tf.stack([offset_width, after_padding_width])
    channel_params = tf.stack([0, 0])
    with tf.control_dependencies([offset_assert]):
        paddings = tf.stack([height_params, width_params, channel_params])
    padded = tf.pad(image, paddings)
    return padded + pad_value


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      The cropped (and resized) image.

    Raises:
      ValueError: if `image` doesn't have rank of 3.
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    if len(image.get_shape().as_list()) != 3:
        raise ValueError('input must have rank of 3')
    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image


def random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.

    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:

      image, depths, normals = random_crop([image, depths, normals], 120, 150)

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the new height.
      crop_width: the new width.

    Returns:
      the image_list with cropped images.

    Raises:
      ValueError: if there are multiple image inputs provided with different size
        or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.

    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.

    Returns:
      A random scale value selected between minimum and maximum value.

    Raises:
      ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.to_float(min_scale_factor)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([1],
                                 minval=min_scale_factor,
                                 maxval=max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, label=None, scale=1.0):
    """Randomly scales image and label.

    Args:
      image: Image with shape [height, width, 3].
      label: Label with shape [height, width, 1].
      scale: The value to scale image and label.

    Returns:
      Scaled image and label.
    """
    # No random scaling if scale == 1.
    if scale == 1.0:
        return image, label
    image_shape = tf.shape(image)
    new_dim = tf.to_int32(tf.to_float([image_shape[0], image_shape[1]]) * scale)

    # Need squeeze and expand_dims because image interpolation takes
    # 4D tensors as input.
    image = tf.squeeze(tf.image.resize_bilinear(
        tf.expand_dims(image, 0),
        new_dim,
        align_corners=True), [0])
    if label is not None:
        label = tf.squeeze(tf.image.resize_nearest_neighbor(
            tf.expand_dims(label, 0),
            new_dim,
            align_corners=True), [0])

    return image, label


def resolve_shape(tensor, rank=None, scope=None):
    """Fully resolves the shape of a Tensor.

    Use as much as possible the shape components already known during graph
    creation and resolve the remaining ones during runtime.

    Args:
      tensor: Input tensor whose shape we query.
      rank: The rank of the tensor, provided that we know it.
      scope: Optional name scope.

    Returns:
      shape: The full shape of the tensor.
    """
    with tf.name_scope(scope, 'resolve_shape', [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()

        if None in shape:
            shape_dynamic = tf.shape(tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]

        return shape


def resize_to_range(image,
                    label=None,
                    min_size=None,
                    max_size=None,
                    factor=None,
                    align_corners=True,
                    label_layout_is_chw=False,
                    scope=None,
                    method=tf.image.ResizeMethod.BILINEAR):
    """Resizes image or label so their sides are within the provided range.

    The output size can be described by two cases:
    1. If the image can be rescaled so its minimum size is equal to min_size
       without the other side exceeding max_size, then do so.
    2. Otherwise, resize so the largest side is equal to max_size.

    An integer in `range(factor)` is added to the computed sides so that the
    final dimensions are multiples of `factor` plus one.

    Args:
      image: A 3D tensor of shape [height, width, channels].
      label: (optional) A 3D tensor of shape [height, width, channels] (default)
        or [channels, height, width] when label_layout_is_chw = True.
      min_size: (scalar) desired size of the smaller image side.
      max_size: (scalar) maximum allowed size of the larger image side. Note
        that the output dimension is no larger than max_size and may be slightly
        smaller than min_size when factor is not None.
      factor: Make output size multiple of factor plus one.
      align_corners: If True, exactly align all 4 corners of input and output.
      label_layout_is_chw: If true, the label has shape [channel, height, width].
        We support this case because for some instance segmentation dataset, the
        instance segmentation is saved as [num_instances, height, width].
      scope: Optional name scope.
      method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.

    Returns:
      A 3-D tensor of shape [new_height, new_width, channels], where the image
      has been resized (with the specified method) so that
      min(new_height, new_width) == ceil(min_size) or
      max(new_height, new_width) == ceil(max_size).

    Raises:
      ValueError: If the image is not a 3D tensor.
    """
    with tf.name_scope(scope, 'resize_to_range', [image]):
        new_tensor_list = []
        min_size = tf.to_float(min_size)
        if max_size is not None:
            max_size = tf.to_float(max_size)
            # Modify the max_size to be a multiple of factor plus 1 and make sure the
            # max dimension after resizing is no larger than max_size.
            if factor is not None:
                max_size = (max_size + (factor - (max_size - 1) % factor) % factor
                            - factor)

        [orig_height, orig_width, _] = resolve_shape(image, rank=3)
        orig_height = tf.to_float(orig_height)
        orig_width = tf.to_float(orig_width)
        orig_min_size = tf.minimum(orig_height, orig_width)

        # Calculate the larger of the possible sizes
        large_scale_factor = min_size / orig_min_size
        large_height = tf.to_int32(tf.ceil(orig_height * large_scale_factor))
        large_width = tf.to_int32(tf.ceil(orig_width * large_scale_factor))
        large_size = tf.stack([large_height, large_width])

        new_size = large_size
        if max_size is not None:
            # Calculate the smaller of the possible sizes, use that if the larger
            # is too big.
            orig_max_size = tf.maximum(orig_height, orig_width)
            small_scale_factor = max_size / orig_max_size
            small_height = tf.to_int32(tf.ceil(orig_height * small_scale_factor))
            small_width = tf.to_int32(tf.ceil(orig_width * small_scale_factor))
            small_size = tf.stack([small_height, small_width])
            new_size = tf.cond(
                tf.to_float(tf.reduce_max(large_size)) > max_size,
                lambda: small_size,
                lambda: large_size)
        # Ensure that both output sides are multiples of factor plus one.
        if factor is not None:
            new_size += (factor - (new_size - 1) % factor) % factor
        new_tensor_list.append(tf.image.resize_images(
            image, new_size, method=method, align_corners=align_corners))
        if label is not None:
            if label_layout_is_chw:
                # Input label has shape [channel, height, width].
                resized_label = tf.expand_dims(label, 3)
                resized_label = tf.image.resize_nearest_neighbor(
                    resized_label, new_size, align_corners=align_corners)
                resized_label = tf.squeeze(resized_label, 3)
            else:
                # Input label has shape [height, width, channel].
                resized_label = tf.image.resize_images(
                    label, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                    align_corners=align_corners)
            new_tensor_list.append(resized_label)
        else:
            new_tensor_list.append(None)
        return new_tensor_list
