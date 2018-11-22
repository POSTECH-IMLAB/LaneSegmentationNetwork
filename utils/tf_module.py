from utils.tf_util import *
from network.resnet_v2 import bottleneck
from tensorflow.contrib.framework.python.ops import arg_scope
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops

FLAGS = tf.app.flags.FLAGS


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    if weights is not None:
        per_entry_cross_ent *= tf.tile(tf.expand_dims(weights, axis=1), [1, 21])
    return tf.reduce_sum(per_entry_cross_ent)


def scale_invariant_feature_extraction(inputs, batch_norm_decay, weight_decay, is_training, feature_depth=256,
                                       output_depth=256):
    with tf.variable_scope("feature_extraction"):
        with arg_scope([conv2d_layer, multi_conv2d_layer, global_avg_pooling_layer], to_batch_norm=True,
                       batch_norm_decay=batch_norm_decay, is_training=is_training, activation_fn=tf.nn.relu,
                       weights_regularizer=slim.l2_regularizer(weight_decay)):
            conv1x1 = conv2d_layer(inputs, feature_depth, 1)
            mul_conv2d = multi_conv2d_layer(inputs, feature_depth, 3, basis_rate=[1, 2, 4, 6, 8])
            global_avg_pooling = global_avg_pooling_layer(inputs, feature_depth, upsample=True)

            concat = tf.concat([conv1x1, mul_conv2d, global_avg_pooling], axis=3)
            output = conv2d_layer(concat, output_depth, 1)

            return output


def scale_invariant_feature_extraction2(inputs, batch_norm_decay, is_training, feature_depth=256, output_depth=256):
    with tf.variable_scope("feature_extraction"):
        with arg_scope([conv2d_layer, multi_conv2d_layer, global_avg_pooling_layer], to_batch_norm=True,
                       batch_norm_decay=batch_norm_decay, is_training=is_training, activation_fn=tf.nn.relu,
                       weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)):
            mul_conv2d = multi_conv2d_layer(inputs, feature_depth, 3, basis_rate=[1, 3, 5, 7, 9])
            global_avg_pooling = global_avg_pooling_layer(inputs, feature_depth, upsample=True)

            concat = tf.concat([mul_conv2d, global_avg_pooling], axis=3)
            output = conv2d_layer(concat, output_depth, 1)

            return output


def slim_decoder(inputs, batch_norm_decay, weight_decay, is_training, feature_depth=256, output_depth=256):
    with tf.variable_scope('slim_decoder'):
        with arg_scope([depthwise_conv2d_layer, conv2d_layer], to_batch_norm=True,
                       batch_norm_decay=batch_norm_decay, is_training=is_training, activation_fn=tf.nn.relu,
                       weights_regularizer=slim.l2_regularizer(weight_decay)):

            net = depthwise_conv2d_layer(inputs, 3)
            net = conv2d_layer(net, feature_depth, 1)
            net = depthwise_conv2d_layer(net, 3)
            net = conv2d_layer(net, output_depth, 1)

            # Alpha
            net = deconv2d_layer(net, int(output_depth / 2), 3, strides=[1, 2, 2, 1])

            return net


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
    """Atrous Spatial Pyramid Pooling.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
        the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      is_training: A boolean denoting whether the input is for training.
      depth: The depth of the ResNet unit output.

    Returns:
      The atrous spatial pyramid pooling output.
    """
    if output_stride not in [8, 16]:
        raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
        atrous_rates = [2 * rate for rate in atrous_rates]

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': 1e-5,
        'scale': True
    }

    with tf.variable_scope("aspp"):
        with arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                features = list()

                conv_1x1 = slim.conv2d(inputs, depth, 1, scope="conv_1x1")
                features.append(conv_1x1)

                # Atrous convolutions
                for a in atrous_rates:
                    conv_3x3 = slim.conv2d(inputs, depth, 3, rate=a, scope='conv_3x3_' + str(a))
                    features.append(conv_3x3)

                with tf.variable_scope("image_level_features"):
                    image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling',
                                                          keepdims=True)
                    image_level_features = slim.conv2d(image_level_features, depth, 1, stride=1,
                                                       scope='conv_1x1')
                    image_level_features = tf.image.resize_bilinear(image_level_features, tf.shape(inputs)[1:3],
                                                                    name='upsample', align_corners=True)
                    features.append(image_level_features)

                net = tf.concat(features, axis=3, name='concat')
                net = slim.conv2d(net, depth, 1, scope='conv_1x1_concat')

                return net


def online_hard_example_mining(logits, one_hot_labels, num_classes, remain_rate=0.3):
    loss_image = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=one_hot_labels,
        logits=tf.reshape(logits, [-1, num_classes])
    )

    logits_shape = tf.shape(logits)
    logits_ = tf.reshape(loss_image, [logits_shape[0], -1])
    k = tf.cast(tf.multiply(remain_rate, tf.cast(logits_shape[1] * logits_shape[2], tf.float32)), tf.int32)
    values, indices = tf.nn.top_k(logits_, k, sorted=False)
    selection = tf.greater_equal(logits_, tf.tile(tf.reduce_min(values, axis=1, keepdims=True),
                                                  [1, tf.shape(logits_)[1]]))
    selection_mask = tf.cast(selection, tf.float32)
    selection = tf.reshape(selection_mask, [-1, 1])
    selection = tf.tile(selection, [1, num_classes])

    loss_logits = tf.reshape(logits, [-1, num_classes]) * selection

    return loss_logits, selection_mask, loss_image


def semantic_confuse_loss(logits, labels, weights, num_classes, alpha=0.5):
    logits_shape = tf.shape(logits)
    softmax_logits = tf.nn.softmax(logits, axis=3)
    log_softmax_logits = -tf.log(softmax_logits)
    max_softmax_logits = tf.reduce_max(softmax_logits * tf.reshape(labels, [-1, logits_shape[1], logits_shape[2], num_classes]),
                                       axis=3, keepdims=True)
    max_softmax_logits = -tf.log(max_softmax_logits)
    filter_size = np.array([25, 25])  # Must be odd
    gaussian_filter_value = gaussian_kernel(int(filter_size[0]/2), int(filter_size[1]/2)) / 2
    gaussian_filter_value = np.reshape(gaussian_filter_value, [filter_size[0], filter_size[1], 1, 1])
    gaussian_filter = tf.constant(
        value=gaussian_filter_value,
        dtype=tf.float32,
        name='gaussian_filter'
    )
    _max_softmax_logits = max_pooling(max_softmax_logits)
    _max_softmax_logits = max_pooling(_max_softmax_logits)
    gaussian_max_softmax_logits = conv2d(_max_softmax_logits, gaussian_filter)
    confusion_weights = tf.clip_by_value(gaussian_max_softmax_logits + alpha, alpha, 1.0)
    confusion_weights = tf.image.resize_bilinear(confusion_weights, tf.shape(logits)[1:3], True)

    # weights = weights * tf.reshape(confusion_weights, [-1])
    # softmax_cross_entropy_loss = tf.reshape(log_softmax_logits, [-1, num_classes]) *\
    #                              tf.reshape(labels, [-1, num_classes])
    # softmax_cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(softmax_cross_entropy_loss, axis=1) * weights)

    return confusion_weights


def class_and_spatial_loss(logits, onehot_labels, weights, weights2):
    logits_shape = tf.shape(logits)
    onehot_labels_shape = tf.shape(onehot_labels)
    image_labels = tf.reshape(onehot_labels, logits_shape)
    class_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=tf.reshape(logits, [-1, onehot_labels_shape[-1]]),
        weights=weights * weights2
    )

    image_weights = tf.reshape(weights, [logits_shape[0], logits_shape[1], logits_shape[2], 1])
    predict_class = tf.argmax(logits, axis=3)
    predict_class = slim.one_hot_encoding(predict_class, onehot_labels_shape[-1], 1.0, 0.0)
    union = to_float(to_bool(predict_class + image_labels)) * image_weights
    intersection = to_float(tf.logical_and(to_bool(predict_class), to_bool(image_labels))) * image_weights
    label_on = to_float(tf.greater(tf.reduce_sum(image_labels, axis=[1, 2]), 0))
    spatial_loss = ((tf.reduce_sum(intersection, axis=[1, 2]) + 1) / (tf.reduce_sum(union, axis=[1, 2]) + 1))
    spatial_loss = tf.reduce_mean(-tf.log(spatial_loss) * label_on)

    return class_loss + spatial_loss


def focal_loss_like(logits, one_hot_labels, num_classes):
    loss_image = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=one_hot_labels,
        logits=tf.reshape(logits, [-1, num_classes])
    )
    logits_shape = tf.shape(logits)
    loss_image = tf.reshape(loss_image, [-1, logits_shape[1], logits_shape[2], 1])

    loss_image = loss_image * loss_image

    focal_loss = tf.reshape(loss_image, [-1])

    return focal_loss


def background_extractor(inputs, batch_norm_decay, is_training):
    num_blocks = [5, 3]
    base_depth = [64, 128]
    unit_rates = [[1] * (num_blocks[0] - 1) + [2], [1, 2, 4]]
    _conv = inputs
    for i, (n, d) in enumerate(zip(num_blocks, base_depth)):
        for j in range(n):
            _conv = bottleneck(_conv, d * 4, d, stride=1, unit_rate=unit_rates[i][j], rate=1)
    sife = scale_invariant_feature_extraction(_conv, batch_norm_decay, is_training)
    compress = conv2d(sife, kernels([1, 1, 256, 2], name='compress_weights'))
    background = compress

    return background


def gaussian_kernel(size, size_y=None, sigma=1.0):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
    g = np.exp(-(x ** 2 / (float(size) * sigma) + y ** 2 / (float(size_y) * sigma)))
    return g / g.sum()


def seg_modify_gradient_weight(labels, remain_rate=0.5, maximum=4.0):
    labels_flat = tf.reshape(labels, [-1])
    one_hot_labels = slim.one_hot_encoding(labels_flat, FLAGS.num_classes, 1.0, 0.0)
    label_size = tf.shape(labels)
    one_hot_label_images = tf.reshape(one_hot_labels, [-1, label_size[1], label_size[2], FLAGS.num_classes])

    filter_size = np.array([35, 35])  # Must be odd

    gaussian_filter_value = gaussian_kernel(int(filter_size[0]/2), int(filter_size[1]/2), sigma=3.0)
    gaussian_filter_value = np.reshape(gaussian_filter_value, [filter_size[0], filter_size[1], 1, 1])
    gaussian_filter = tf.constant(
        value=gaussian_filter_value,
        dtype=tf.float32,
        name='gaussian_filter'
    )

    edge_check_filter_value = np.array(
        [
            [1, -0.5],
            [-0.5, 0]
        ]
    )
    edge_check_filter = tf.constant(
        value=edge_check_filter_value,
        dtype=tf.float32,
        name='edge_filter'
    )
    edge_check_filter = tf.reshape(edge_check_filter, [2, 2, 1, 1])
    edge_check_filter = tf.tile(edge_check_filter, [1, 1, FLAGS.num_classes, 1])
    padded_label = tf.pad(one_hot_label_images, [[0, 0], [0, 1], [0, 1], [0, 0]], mode='SYMMETRIC')
    padded_label = tf.cast(padded_label, tf.float32)

    edge_image = tf.nn.depthwise_conv2d(padded_label, edge_check_filter, strides=[1, 1, 1, 1], padding='VALID')
    edge_image = tf.cast(tf.not_equal(edge_image, 0), tf.float32)
    label_size = tf.shape(labels)

    compress_filter_value = np.array(
        [
            [1, 1],
            [1, 1]
        ]
    )
    compress_filter = tf.constant(
        value=compress_filter_value,
        dtype=tf.float32,
        name='compress_filter'
    )
    compress_filter = tf.reshape(compress_filter, [2, 2, 1, 1])
    edge_image = tf.reduce_sum(edge_image, axis=3, keepdims=True)
    edge_image = conv2d(edge_image, compress_filter, strides=[1, 2, 2, 1])
    edge_image = conv2d(edge_image, compress_filter, strides=[1, 2, 2, 1])

    gaussian_edge = conv2d(edge_image, gaussian_filter)
    gaussian_edge = tf.image.resize_bilinear(gaussian_edge, size=label_size[1:3], align_corners=True)
    label_weights = tf.clip_by_value(gaussian_edge * 1.5 + remain_rate,
                                     clip_value_min=0.0, clip_value_max=maximum)

    return label_weights, gaussian_edge


def semantic_super_pixel_loss(inputs, labels, one_hot_labels, not_ignore_mask):
    """
    * Overlap loss, 안 겹치게: min(reduce_sum_channel(x) - 1)
    * Edge loss, Edge에서 액티베이션 덜 되게: min(reduce_sum_channel(x) * edge_weights)
    * Bias loss, 안 치우치게: min(max(reduce_sum_width_height(x)) - w * h / c * alpha)
    * Semantic Super Pixel Loss = Overlap loss + Edge loss + Bias loss
    :param inputs:
    :param labels:
    :param weights
    :param output_channel
    :return:
    """
    with tf.variable_scope('ssp_loss', reuse=tf.AUTO_REUSE):
        input_shape = inputs.get_shape()
        label_shape = labels.get_shape()
        output_channel = one_hot_labels.get_shape()[-1]

        gaussian_filter_size = [15, 15]
        gaussian_filter_value = gaussian_kernel(int(gaussian_filter_size[0] / 2), int(gaussian_filter_size[1] / 2))
        gaussian_filter_value = gaussian_filter_value * 2 -\
                                gaussian_filter_value[int(gaussian_filter_size[0]/2)][int(gaussian_filter_size[1]/2)] / 2
        gaussian_filter_value = np.reshape(gaussian_filter_value, [gaussian_filter_size[0], gaussian_filter_size[1],
                                                                   1, 1])
        gaussian_filters = tf.constant(
            value=gaussian_filter_value,
            dtype=tf.float32,
            name='gaussian_filter'
        )
        gaussian_filters = tf.tile(gaussian_filters, [1, 1, input_shape[3], 1])
        cohesion = tf.nn.depthwise_conv2d(tf.sigmoid(inputs), filter=gaussian_filters, strides=[1, 1, 1, 1], padding='SAME')
        cohesion_loss = (tf.sigmoid(inputs) - tf.sigmoid(cohesion))
        cohesion_loss = tf.reduce_mean(cohesion_loss, axis=3, keepdims=True)
        cohesion_loss = tf.image.resize_bilinear(cohesion_loss, tf.shape(labels)[1:3], True)
        cohesion_loss = tf.reduce_mean(tf.reshape(cohesion_loss, [-1]) * not_ignore_mask)

        # activated_inputs = tf.cast(tf.greater(inputs, 0), tf.float32)
        # overlap_bias = tf.cast(tf.shape(inputs)[1] * tf.shape(inputs)[2], tf.float32) / tf.cast(label_shape[-1],
        #                                                                                         tf.float32)
        # activated_inputs = tf.reduce_sum(activated_inputs, [1, 2], keepdims=True)
        # activated_inputs = tf.image.resize_bilinear(activated_inputs, label_shape[1:3], True)
        # average_loss = tf.reduce_mean(tf.pow(tf.reshape(overlap_bias - activated_inputs, [-1]) * not_ignore_mask, 2))

        weights = kernels([input_shape[3], output_channel],
                          initializer=tf.random_uniform_initializer(0, 1),
                          regularizer=slim.l2_regularizer(FLAGS.weight_decay),
                          name='weights')
        weights = tf.nn.softmax(weights * 100, axis=1)
        weights = tf.reshape(weights, [1, 1, input_shape[3], output_channel])
        output_selection = conv2d(tf.nn.sigmoid(inputs), weights)
        output_selection = tf.image.resize_bilinear(output_selection, tf.shape(labels)[1:3], True)
        activated_labels = tf.reduce_sum(tf.reshape(one_hot_labels,
                                                    [-1, label_shape[1], label_shape[2], output_channel])
                                         , axis=[1, 2], keepdims=True)
        activated_labels = tf.cast(tf.greater(activated_labels, 0), tf.float32)
        activated_labels = tf.image.resize_nearest_neighbor(activated_labels, tf.shape(labels)[1:3], True)

        out_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_labels,
            logits=tf.reshape(output_selection, [-1, output_channel])
        )
        out_loss = tf.reshape(out_loss, [-1])
        out_loss = out_loss * not_ignore_mask
        out_loss = tf.reduce_mean(out_loss)

        ssp_loss = out_loss  # + average_loss

        for i in range(10):
            tf.summary.image('ssp_tensors', tf.expand_dims(tf.transpose(inputs, [3, 0, 1, 2])[i], axis=3),
                             max_outputs=1)

        tf.summary.scalar('ssp_loss', tf.reduce_mean(ssp_loss))
        tf.summary.scalar('out_loss', tf.reduce_mean(out_loss))
        tf.summary.scalar('cohesion_loss', tf.reduce_mean(cohesion_loss))
        # tf.summary.scalar('average_loss', tf.reduce_mean(average_loss))

    return ssp_loss


