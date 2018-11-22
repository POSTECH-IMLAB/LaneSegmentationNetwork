import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.framework import add_arg_scope
import tensorflow.contrib.slim as slim
from tensorflow.contrib.nccl.ops import gen_nccl_ops
from tensorflow.contrib.framework import add_model_variable


def to_bool(tensor):
    return tf.cast(tensor, tf.bool)


def to_float(tensor):
    return tf.cast(tensor, tf.float32)


def get_conv_kernel(inputs, output_channel, kernel_size, initializer=None, weights_regularizer=None, name=None,
                    is_transpose=False):
    input_shape = inputs.get_shape()
    if type(kernel_size) is int:
        if is_transpose:
            kernel = kernels([kernel_size, kernel_size, output_channel, input_shape[3]], initializer=initializer,
                             regularizer=weights_regularizer, name=name)
        else:
            kernel = kernels([kernel_size, kernel_size, input_shape[3], output_channel], initializer=initializer,
                             regularizer=weights_regularizer, name=name)
    elif type(kernel_size) is list and len(kernel_size) == 2:
        if is_transpose:
            kernel = kernels([kernel_size[0], kernel_size[1], output_channel, input_shape[3]],
                             initializer=initializer, regularizer=weights_regularizer, name=name)
        else:
            kernel = kernels([kernel_size[0], kernel_size[1], input_shape[3], output_channel],
                             initializer=initializer, regularizer=weights_regularizer, name=name)
    else:
        raise ValueError('filter_size must be an integer or list of length 2.')

    return kernel


def get_bias(inputs, initializer=None, name=None):
    input_shape = inputs.get_shape()
    if initializer is not None:
        bias = tf.get_variable(name, [1, input_shape[1], input_shape[2], input_shape[3]],
                               initializer=initializer)
    else:
        bias = None
    return bias


@add_arg_scope
def batch_norm(x, is_training, decay=0.997, epsilon=1e-5, scale=True, center=True):
    normed = layers.batch_norm(
        x,
        decay=decay,
        epsilon=epsilon,
        is_training=is_training,
        center=center,
        scale=scale
    )
    return normed


@add_arg_scope
def conv2d(
        inputs, filters, bias=None,
        strides=list([1, 1, 1, 1]), padding='SAME', dilations=list([1, 1, 1, 1]),
        to_batch_norm=False, batch_norm_decay=0.997, is_training=True, activation_fn=None, name=None
):
    output = tf.nn.conv2d(
        input=inputs,
        filter=filters,
        strides=strides,
        padding=padding,
        dilations=dilations,
        name=name
    )

    if bias is not None:
        output = tf.nn.bias_add(output, bias)
    if to_batch_norm:
        output = batch_norm(output, is_training, batch_norm_decay)
    if activation_fn is not None:
        output = activation_fn(output)
    return output


@add_arg_scope
def conv2d_layer(inputs, output_channel, kernel_size, bias_initializer=None, weights_regularizer=None,
                 strides=list([1, 1, 1, 1]), padding='SAME', dilations=list([1, 1, 1, 1]),
                 to_batch_norm=False, batch_norm_decay=0.997, is_training=True, activation_fn=None, scope=None,
                 return_weights=False):
    with tf.variable_scope(scope, 'conv2d'):
        kernel = get_conv_kernel(inputs, output_channel, kernel_size, weights_regularizer=weights_regularizer,
                                 name='weights')
        bias = get_bias(inputs, bias_initializer, 'bias')

        output = conv2d(inputs, kernel, bias, strides, padding, dilations, to_batch_norm, batch_norm_decay,
                        is_training, activation_fn, name='conv')

        if return_weights:
            return output, kernel, bias
        else:
            return output


@add_arg_scope
def deconv2d(
        inputs, filters, bias=None, strides=list([1, 1, 1, 1]), padding='SAME', to_batch_norm=False,
        batch_norm_decay=0.997, is_training=True, activation_fn=None, name=None):
    input_shape = inputs.get_shape().as_list()
    filter_shape = filters.get_shape().as_list()
    if padding in ['SAME', 'same']:
        output_shape = [
            tf.shape(inputs)[0],
            input_shape[1] * strides[1],
            input_shape[2] * strides[2],
            filter_shape[2]
        ]
    elif padding == ['VALID', 'valid']:
        output_shape = [
            tf.shape(inputs)[0],
            (input_shape[1] - 1) * strides[1] + filter_shape[0],
            (input_shape[2] - 1) * strides[2] + filter_shape[1],
            filter_shape[2]
        ]
    else:
        output_shape = [
            tf.shape(inputs)[0],
            input_shape[1] * strides[1],
            input_shape[2] * strides[2],
            filter_shape[2]
        ]

    output = tf.nn.conv2d_transpose(
        value=inputs,
        filter=filters,
        output_shape=output_shape,
        strides=strides,
        padding=padding,
        name=name
    )

    if bias is not None:
        output = tf.nn.bias_add(output, bias)

    if to_batch_norm:
        output = batch_norm(output, is_training, batch_norm_decay)

    if activation_fn is not None:
        output = activation_fn(output)

    return output


@add_arg_scope
def deconv2d_layer(inputs, output_channel, kernel_size, bias_initializer=None, weights_regularizer=None,
                   strides=list([1, 1, 1, 1]), padding='SAME', to_batch_norm=False, batch_norm_decay=0.997,
                   is_training=True, activation_fn=None, scope=None, return_weights=False):
    with tf.variable_scope(scope, 'deconv2d'):
        kernel = get_conv_kernel(inputs, output_channel, kernel_size, weights_regularizer=weights_regularizer,
                                 name='weights', is_transpose=True)
        bias = get_bias(inputs, bias_initializer, 'bias')

        output = deconv2d(inputs, kernel, bias, strides, padding, to_batch_norm, batch_norm_decay,
                          is_training, activation_fn, name='conv')

        if return_weights:
            return output, kernel, bias
        else:
            return output


@add_arg_scope
def depthwise_conv2d(
        inputs, filters, bias=None,
        strides=list([1, 1, 1, 1]), padding='SAME', dilations=list([1, 1, 1, 1]),
        to_batch_norm=False, batch_norm_decay=0.997, is_training=True, activation_fn=None, name=None
):
    if isinstance(strides, int):
        strides = list([1, strides, strides, 1])
    if isinstance(dilations, int):
        dilations = list([1, dilations, dilations, 1])

    output = tf.nn.depthwise_conv2d(
        input=inputs,
        filter=filters,
        strides=strides,
        padding=padding,
        rate=dilations,
        name=name
    )

    if bias is not None:
        output = tf.nn.bias_add(output, bias)
    if to_batch_norm:
        output = batch_norm(output, is_training, batch_norm_decay)
    if activation_fn is not None:
        output = activation_fn(output)
    return output


@add_arg_scope
def depthwise_conv2d_layer(inputs, kernel_size, bias_initializer=None, weights_regularizer=None,
                           strides=list([1, 1, 1, 1]), padding='SAME', dilations=list([1, 1, 1, 1]),
                           to_batch_norm=False, batch_norm_decay=0.997, is_training=True, activation_fn=None, scope=None,
                           return_weights=False):
    with tf.variable_scope(scope, 'conv2d'):
        kernel = get_conv_kernel(inputs, 1, kernel_size, weights_regularizer=weights_regularizer,
                                 name='weights')
        bias = get_bias(inputs, bias_initializer, 'bias')

        output = depthwise_conv2d(inputs, kernel, bias, strides, padding, dilations, to_batch_norm, batch_norm_decay,
                                  is_training, activation_fn, name='conv')

        if return_weights:
            return output, kernel, bias
        else:
            return output


@add_arg_scope
def atrous_conv2d(inputs, filters, rate, bias=None, padding='SAME',
                  stride=list([1, 1, 1, 1]), to_batch_norm=False, batch_norm_decay=0.997, is_training=True,
                  activation_fn=None, name=None):
    if rate <= 0:
        raise ValueError('Rate of atrous_conv2d must be larger or equal than 0')
    if type(rate) is not int:
        raise ValueError('Rate of atrous_conv2d must be integer value')

    output = tf.nn.conv2d(
        input=inputs,
        filter=filters,
        strides=stride,
        padding=padding,
        dilations=[1, rate, rate, 1],
        name=name
    )

    if bias is not None:
        output = tf.nn.bias_add(output, bias)
    if to_batch_norm:
        output = batch_norm(output, is_training, batch_norm_decay)
    if activation_fn is not None:
        output = activation_fn(output)

    return output


@add_arg_scope
def bilinear_conv2d(inputs, filters, rescale: float, bias=None, padding='SAME', stride=list([1, 1, 1, 1]),
                    to_batch_norm=False, batch_norm_decay=0.997, is_training=True,
                    activation_fn=None, name=None):

    if rescale != 1.0:
        input_size = tf.shape(inputs)[1:3]
        resized_input = tf.image.resize_bilinear(
            inputs,
            tf.cast(
                tf.round(
                    tf.cast(input_size, tf.float32) / rescale + tf.constant(1, dtype=tf.float32, shape=[2])
                ),
                tf.int32
            ),
            align_corners=True
        )

        output = conv2d(
            inputs=resized_input,
            filters=filters,
            strides=stride,
            padding=padding,
            name=name
        )

        output = tf.image.resize_bilinear(
            output,
            tf.cast(
                tf.ceil(
                    tf.cast(input_size, tf.float32) / np.array(stride[1:3])
                ),
                tf.int32
            ),
            align_corners=True
        )
    else:
        output = conv2d(
            inputs=inputs,
            filters=filters,
            strides=stride,
            padding=padding,
            name=name
        )

    if bias is not None:
        output = tf.nn.bias_add(output, bias)
    if to_batch_norm:
        output = batch_norm(output, is_training, batch_norm_decay)
    if activation_fn is not None:
        output = activation_fn(output)

    return output


@add_arg_scope
def multi_conv2d(inputs, filters: tf.Tensor, bias=None, stride=list([1, 1, 1, 1]),
                 padding='SAME', basis_rate=list([1, 3, 5]), to_batch_norm=False, batch_norm_decay=0.997,
                 is_training=True, activation_fn=None):
    _number_of_basis = len(basis_rate)
    if _number_of_basis < 2:
        raise ValueError('Number of basis_rate must be larger or equal than 2')

    output = conv2d(inputs, filters, bias, stride, padding)
    for idx, r in enumerate(basis_rate):
        output += atrous_conv2d(inputs, filters, r, bias, padding, stride)
    output /= _number_of_basis

    if to_batch_norm:
        output = batch_norm(output, is_training, batch_norm_decay)

    if activation_fn is not None:
        output = activation_fn(output)

    return output


@add_arg_scope
def multi_conv2d_modified(inputs, filters: tf.Tensor, bias=None,
                          stride=list([1, 1, 1, 1]), padding='SAME', basis_rate=list([1, 3, 5]),
                          to_batch_norm=False, batch_norm_decay=0.997, is_training=True, activation_fn=None):
    _number_of_basis = len(basis_rate)
    # _filter_shape = tf.shape(filters)
    # _filter_center = tf.slice(filters, [1, 1, 0, 0], [1, 1, _filter_shape[2], _filter_shape[3]])

    if _number_of_basis < 2:
        raise ValueError('Number of basis_rate must be larger or equal than 2')

    input_shape = inputs.get_shape()
    output_channel = filters.get_shape()[-1]
    global_average_pooling = global_avg_pooling_layer(inputs, upsample=False)
    depth = 256
    selection_weights1 = kernels([input_shape[-1], depth],
                                 regularizer=slim.l2_regularizer(0.0001),
                                 name='rate_selection_weights1')
    selection_weights2 = kernels([depth, _number_of_basis],
                                 regularizer=slim.l2_regularizer(0.0001),
                                 name='rate_selection_weights2')

    global_avg_pooling_squeezed = tf.squeeze(global_average_pooling, axis=[1, 2])

    selection = tf.matmul(global_avg_pooling_squeezed, selection_weights1)
    selection = batch_norm(selection, is_training, batch_norm_decay)
    selection = tf.nn.relu(selection)

    selection = tf.matmul(selection, selection_weights2)
    selection = batch_norm(selection, is_training, batch_norm_decay)
    selection = tf.nn.relu(selection)

    selection = tf.transpose(selection, [1, 0])
    output = None
    for idx, r in enumerate(basis_rate):
        if idx == 0:
            output = tf.einsum('nhwc,n->nhwc', atrous_conv2d(inputs, filters, r, bias, padding, stride), selection[idx])
        output += tf.einsum('nhwc,n->nhwc', atrous_conv2d(inputs, filters, r, bias, padding, stride), selection[idx])

    if to_batch_norm:
        output = batch_norm(output, is_training, batch_norm_decay)

    if activation_fn is not None:
        output = activation_fn(output)

    return output


@add_arg_scope
def multi_conv2d_modified2(inputs, filters: tf.Tensor, bias=None,
                           stride=list([1, 1, 1, 1]), padding='SAME', basis_rate=list([1, 3, 5]),
                           to_batch_norm=False, batch_norm_decay=0.997, is_training=True, activation_fn=None):
    _number_of_basis = len(basis_rate)
    # _filter_shape = tf.shape(filters)
    # _filter_center = tf.slice(filters, [1, 1, 0, 0], [1, 1, _filter_shape[2], _filter_shape[3]])

    if _number_of_basis < 2:
        raise ValueError('Number of basis_rate must be larger or equal than 2')

    input_shape = inputs.get_shape()
    output_channel = filters.get_shape()[-1]
    global_average_pooling = global_avg_pooling_layer(inputs, upsample=False)
    depth = 256
    selection_weights1 = kernels([input_shape[-1], depth],
                                 regularizer=slim.l2_regularizer(0.0001),
                                 name='rate_selection_weights1')
    selection_weights2 = kernels([depth, _number_of_basis],
                                 regularizer=slim.l2_regularizer(0.0001),
                                 name='rate_selection_weights2')

    global_avg_pooling_squeezed = tf.squeeze(global_average_pooling, axis=[1, 2])

    selection = tf.matmul(global_avg_pooling_squeezed, selection_weights1)
    selection = batch_norm(selection, is_training, batch_norm_decay)
    selection = tf.nn.relu(selection)

    selection = tf.matmul(selection, selection_weights2)
    selection = batch_norm(selection, is_training, batch_norm_decay)
    selection = tf.nn.relu(selection)

    selection = tf.transpose(selection, [1, 0])
    output = list()
    for idx in range(8):
        input_selection = tf.expand_dims(inputs[idx], axis=0)
        _result = atrous_conv2d(input_selection, filters, basis_rate[0], bias, padding, stride)
        for r in basis_rate[1:]:
            if idx == 0:
                output = atrous_conv2d(input_selection, filters, r, bias, padding, stride)
            output += atrous_conv2d(input_selection, filters, r, bias, padding, stride)

    if to_batch_norm:
        output = batch_norm(output, is_training, batch_norm_decay)

    if activation_fn is not None:
        output = activation_fn(output)

    return output


@add_arg_scope
def multi_conv2d_layer(inputs, output_channel, kernel_size, bias_initializer=None,
                       stride=list([1, 1, 1, 1]), padding='SAME', basis_rate=list([1, 3, 5]), weights_regularizer=None,
                       to_batch_norm=False, batch_norm_decay=0.997, is_training=True, activation_fn=None,
                       scope=None, return_weights=False):
    with tf.variable_scope(scope, 'multi_conv2d'):
        kernel = get_conv_kernel(inputs, output_channel, kernel_size, weights_regularizer=weights_regularizer,
                                 name='weights')
        bias = get_bias(inputs, bias_initializer, 'bias')

        output = multi_conv2d(inputs, kernel, bias, stride, padding, basis_rate,
                              to_batch_norm, batch_norm_decay, is_training, activation_fn)

        if return_weights:
            return output, kernel, bias
        else:
            return output


@add_arg_scope
def scale_conv2d(inputs, filters: tf.Tensor, bias=None, stride=list([1, 1, 1, 1]), padding='SAME',
                 initial_step=1, number_of_step=5, step_multiplier=1.25,
                 to_batch_norm=False, batch_norm_decay=0.997, is_training=True, activation_fn=None):
    _step = initial_step
    output = bilinear_conv2d(inputs, filters, _step, bias, padding, stride)

    for i in range(1, number_of_step):
        _step *= step_multiplier
        output += bilinear_conv2d(inputs, filters, _step, bias, padding, stride)
    output /= number_of_step

    if to_batch_norm:
        output = batch_norm(output, is_training, batch_norm_decay)

    if activation_fn is not None:
        output = activation_fn(output)

    return output


@add_arg_scope
def scale_conv2d_layer(inputs, output_channel, kernel_size, bias_initializer=None, stride=list([1, 1, 1, 1]),
                       weights_regularizer=None, padding='SAME', initial_step=1, number_of_step=5, step_multiplier=1.25,
                       to_batch_norm=False, batch_norm_decay=0.997, is_training=True, activation_fn=None, scope=None,
                       return_weights=False):
    with tf.variable_scope(scope, default_name='scale_conv2d'):
        kernel = get_conv_kernel(inputs, output_channel, kernel_size, weights_regularizer=weights_regularizer,
                                 name='weights')
        bias = get_bias(inputs, bias_initializer, 'bias')

        output = scale_conv2d(inputs, kernel, bias, stride, padding, initial_step, number_of_step, step_multiplier,
                              to_batch_norm, batch_norm_decay, is_training, activation_fn)

        if return_weights:
            return output, kernel, bias
        else:
            return output


@add_arg_scope
def global_avg_pooling_layer(inputs, depth=None, to_batch_norm=False, batch_norm_decay=0.997, is_training=True,
                             activation_fn=None, weights_regularizer=None, upsample=True):
    if depth is None:
        output = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
        if upsample:
            output = tf.image.resize_bilinear(output, tf.shape(inputs)[1:3], name='upsample', align_corners=True)
        if to_batch_norm:
            output = batch_norm(output, is_training, batch_norm_decay)
        if activation_fn:
            output = activation_fn(output)
    else:
        output = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
        output = conv2d_layer(output, depth, 1, to_batch_norm=to_batch_norm, batch_norm_decay=batch_norm_decay,
                              is_training=is_training, activation_fn=activation_fn,
                              weights_regularizer=weights_regularizer)
        if upsample:
            output = tf.image.resize_bilinear(output, tf.shape(inputs)[1:3], name='upsample', align_corners=True)

    return output


@add_arg_scope
def kernels(shape, value=None, initializer=None, stddev=None, trainable=True, regularizer=None,
            dtype=tf.float32, name=None):
    if value is None:
        if stddev is None:
            if initializer is None:
                filters = tf.get_variable(
                    shape=shape,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=regularizer,
                    trainable=trainable,
                    name=name
                )
            else:
                filters = tf.get_variable(
                    shape=shape,
                    initializer=initializer,
                    regularizer=regularizer,
                    trainable=trainable,
                    name=name
                )
        else:
            filters = tf.get_variable(
                shape=shape,
                initializer=tf.random_normal_initializer(stddev=stddev),
                regularizer=regularizer,
                trainable=trainable,
                name=name
            )
    else:
        filters = tf.get_variable(
            initializer=tf.constant_initializer(
                tf.constant(
                    value=value,
                    shape=shape,
                    dtype=dtype
                )
            ),
            regularizer=regularizer,
            trainable=trainable,
            name=name
        )
    return filters


def max_pooling(inputs, kernel_size=list([1, 2, 2, 1]), strides=list([1, 2, 2, 1]), padding='SAME'):
    output = tf.nn.max_pool(
        value=inputs,
        ksize=kernel_size,
        strides=strides,
        padding=padding
    )
    return output


def avg_pooling(inputs, kernel_size=list([1, 2, 2, 1]), strides=list([1, 2, 2, 1]), padding='SAME'):
    output = tf.nn.avg_pool(
        value=inputs,
        ksize=kernel_size,
        strides=strides,
        padding=padding
    )
    return output


def tensor_histogram(tensor, name):
    tf.summary.histogram(name, tensor.value())


@add_arg_scope
def sync_batch_norm(inputs,
                    decay=0.999,
                    epsilon=0.001,
                    scale=False,
                    activation_fn=None,
                    updates_collections=tf.GraphKeys.UPDATE_OPS,
                    is_training=True,
                    reuse=None,
                    variables_collections=None,
                    trainable=True,
                    scope=None,
                    num_dev=1):
    '''
    num_dev is how many gpus you use.
    '''

    red_axises = [0, 1, 2]
    num_outputs = inputs.get_shape().as_list()[-1]

    if scope is None:
        scope = 'BatchNorm'

    with tf.variable_scope(
            scope,
            'BatchNorm',
            reuse=reuse):
        if scale:
            gamma = tf.get_variable(name='gamma', shape=[num_outputs], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=trainable,
                                    collections=variables_collections)
        else:
            gamma = None

        beta = tf.get_variable(name='beta', shape=[num_outputs], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), trainable=trainable,
                               collections=variables_collections)

        moving_mean = tf.get_variable(name='moving_mean', shape=[num_outputs], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0), trainable=False,
                                      collections=variables_collections)

        moving_var = tf.get_variable(name='moving_variance', shape=[num_outputs], dtype=tf.float32,
                                     initializer=tf.constant_initializer(1.0), trainable=False,
                                     collections=variables_collections)

        if is_training and trainable:

            if num_dev == 1:
                mean, var = tf.nn.moments(inputs, red_axises)
            else:
                shared_name = tf.get_variable_scope().name
                batch_mean = tf.reduce_mean(inputs, axis=red_axises)
                batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axises)
                batch_mean = gen_nccl_ops.nccl_all_reduce(
                    input=batch_mean,
                    reduction='sum',
                    num_devices=num_dev,
                    shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
                batch_mean_square = gen_nccl_ops.nccl_all_reduce(
                    input=batch_mean_square,
                    reduction='sum',
                    num_devices=num_dev,
                    shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
                mean = batch_mean
                var = batch_mean_square - tf.square(batch_mean)
            outputs = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)

            if int(outputs.device[-1]) == 0:
                update_moving_mean_op = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
                update_moving_var_op = tf.assign(moving_var, moving_var * decay + var * (1 - decay))
                add_model_variable(moving_mean)
                add_model_variable(moving_var)

                if updates_collections is None:
                    with tf.control_dependencies([update_moving_mean_op, update_moving_var_op]):
                        outputs = tf.identity(outputs)
                else:
                    tf.add_to_collections(updates_collections, update_moving_mean_op)
                    tf.add_to_collections(updates_collections, update_moving_var_op)
                    outputs = tf.identity(outputs)
            else:
                outputs = tf.identity(outputs)

        else:
            outputs, _, _ = tf.nn.fused_batch_norm(inputs, gamma, beta, mean=moving_mean,
                                                   variance=moving_var, epsilon=epsilon, is_training=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def split_tensor(inputs, num_vertical_split, num_horizontal_split):
    transposed_image = tf.transpose(inputs, [0, 3, 1, 2])
    input_shape = tf.shape(transposed_image)
    reshape_image = tf.reshape(transposed_image,
                               [
                                   input_shape[0],
                                   input_shape[1],
                                   num_vertical_split,
                                   tf.cast(input_shape[2] / num_vertical_split, tf.int32),
                                   input_shape[3]
                               ])
    reshape_image = tf.transpose(reshape_image, [0, 2, 1, 4, 3])
    reshape_image = tf.reshape(reshape_image,
                               [
                                   input_shape[0], num_vertical_split * input_shape[1],
                                   input_shape[3], tf.cast(input_shape[2] / num_vertical_split, tf.int32)
                               ])
    reshape_image = tf.reshape(reshape_image,
                               [
                                   input_shape[0],
                                   num_vertical_split * input_shape[1],
                                   num_horizontal_split,
                                   tf.cast(input_shape[3] / num_horizontal_split, tf.int32),
                                   tf.cast(input_shape[2] / num_vertical_split, tf.int32)
                               ])
    result_image = tf.transpose(reshape_image, [0, 4, 3, 2, 1])
    result_image = tf.reshape(result_image,
                              [
                                  input_shape[0],
                                  tf.cast(input_shape[2] / num_vertical_split, tf.int32),
                                  tf.cast(input_shape[3] / num_horizontal_split, tf.int32),
                                  input_shape[1] * num_horizontal_split * num_vertical_split
                              ])

    return result_image
