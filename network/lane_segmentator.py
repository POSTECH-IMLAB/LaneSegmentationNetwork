from utils.tf_util import *
from utils.preprocessing import *
from utils.tf_module import *
from network.resnet_v2 import resnet_v2_50, resnet_v2_101, resnet_v2_101_multi, resnet_arg_scope
from network.resnet_v1_beta import resnet_v1_101_beta
from tensorflow.contrib import slim
from matplotlib import pyplot as plt


FLAGS = tf.app.flags.FLAGS
_MEAN_RGB = [123.15, 115.90, 103.06]


class Segmentator:
    def __init__(self, params: dict, multi_scale_eval=None, flipped_eval=None):
        self.network_name = 'Segmentator_res'
        print('[Load %s]' % self.network_name)

        print('Backbone is', params['base_architecture'])
        self.params = params
        self.model = self.network
        self.multi_scale_eval = multi_scale_eval
        self.flipped_eval = flipped_eval

    def _preprocess_subtract_imagenet_mean(self, inputs):
        """Subtract Imagenet mean RGB value."""
        mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
        return inputs - mean_rgb

    def network(self, inputs, is_training):
        params = self.params

        with tf.name_scope('backbone'):
            if params['base_architecture'] == 'resnet_v2_50':
                base_model = resnet_v2_50
            elif params['base_architecture'] == 'resnet_v2_101':
                base_model = resnet_v2_101
            elif params['base_architecture'] == 'resnet_v1_101':
                base_model = resnet_v1_101_beta
            else:
                raise ValueError('Base architecture must be resnet_v2_50 or resnet_v2_101')

            preprocessed_inputs = self._preprocess_subtract_imagenet_mean(inputs)

            with tf.contrib.slim.arg_scope(resnet_arg_scope(
                    weight_decay=params['weight_decay'],
                    batch_norm_decay=0.95,
                    batch_norm_epsilon=1e-5,
                    batch_norm_scale=True)):
                logits, end_points, auxiliary = base_model(
                    inputs=preprocessed_inputs,
                    num_classes=None,
                    is_training=(is_training and FLAGS.fine_tune_batch_norm),
                    global_pool=False,
                    output_stride=params['output_stride'],
                    multi_grid=[1, 2, 4],
                    reuse=tf.AUTO_REUSE
                )

        # Load pretrained variables
        if is_training:
            if params['pre_trained_model'] is not None:
                exclude = [
                    params['base_architecture'] + '/logits',
                    'global_step'
                ]
                variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
                variables_to_restore = [
                    v for v in variables_to_restore
                    if params['base_architecture'] in v.name.split(':')[0] and 'network/' not in v.name.split(':')[0]
                ]

                tf.train.init_from_checkpoint(
                    ckpt_dir_or_file=params['pre_trained_model'],
                    assignment_map={v.name.split(':')[0]: v for v in variables_to_restore}
                )

        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            features = scale_invariant_feature_extraction(
                inputs=logits,
                weight_decay=params['weight_decay'],
                batch_norm_decay=params['batch_norm_decay'],
                is_training=is_training and FLAGS.fine_tune_batch_norm,
                feature_depth=256,
                output_depth=256
            )

            features_drop_out = slim.dropout(features, 0.9, is_training=is_training)

            rescaled_logits = tf.image.resize_bilinear(features_drop_out, size=tf.to_int32(tf.shape(inputs)[1:3] / 4),
                                                       align_corners=True)
            raw_feature = conv2d_layer(auxiliary, 48, 1,
                                       weights_regularizer=slim.l2_regularizer(params['weight_decay']),
                                       to_batch_norm=True, batch_norm_decay=params['batch_norm_decay'],
                                       is_training=is_training and FLAGS.fine_tune_batch_norm,
                                       activation_fn=tf.nn.relu)

            decoded = slim_decoder(
                inputs=tf.concat([rescaled_logits, raw_feature], axis=3),
                batch_norm_decay=params['batch_norm_decay'],
                weight_decay=params['weight_decay'],
                is_training=is_training and FLAGS.fine_tune_batch_norm,
                feature_depth=256,
                output_depth=256
            )
            decoded = conv2d_layer(decoded, params['num_classes'], 1,
                                   weights_regularizer=slim.l2_regularizer(params['weight_decay']),
                                   to_batch_norm=False, activation_fn=None, scope='conv_merge')

            rescaled_logits = tf.image.resize_bilinear(decoded, size=tf.shape(inputs)[1:3], align_corners=True)

        return rescaled_logits

    def model_fn(self, features, labels, mode):
        params = self.params

        if isinstance(features, dict):
            features = features['feature']

        images = tf.cast(
            features,
            tf.uint8
        )

        logits = self.model(
            inputs=features,
            is_training=(mode == tf.estimator.ModeKeys.TRAIN)
        )

        predict_classes = tf.expand_dims(
            tf.argmax(logits, axis=3, output_type=tf.int32),
            axis=3
        )

        predict_decoded_labels = tf.py_func(
            decode_labels,
            [predict_classes, params['batch_size'], params['num_classes']],
            tf.uint8
        )

        predictions = {
            'classes': predict_classes,
            'probabilities': tf.nn.softmax(logits, name='probabilities'),
            'decoded_labels': predict_decoded_labels
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
            predictions_without_decoded_labels = predictions.copy()
            del predictions_without_decoded_labels['decoded_labels']

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'preds': tf.estimator.export.PredictOutput(
                        predictions_without_decoded_labels)
                })

        gt_decoded_labels = tf.py_func(
            decode_labels,
            [labels, params['batch_size'], params['num_classes']],
            tf.uint8
        )

        labels_flat = tf.reshape(labels, [-1])
        preds_flat = tf.reshape(predict_classes, [-1])

        not_ignore_mask = tf.to_float(tf.not_equal(labels_flat, params['ignore_label']))
        labels_flat = tf.where(
            tf.equal(labels_flat, params['ignore_label']), tf.zeros_like(labels_flat), labels_flat)

        confusion_matrix = tf.confusion_matrix(
            labels_flat, preds_flat,
            num_classes=params['num_classes'],
            weights=not_ignore_mask
        )

        predictions['valid_preds'] = preds_flat
        predictions['valid_labels'] = labels_flat
        predictions['confusion_matrix'] = confusion_matrix

        if not params['fine_tune_batch_norm']:
            train_var_list = [v for v in tf.trainable_variables()]
        else:
            train_var_list = [v for v in tf.trainable_variables()
                              if 'beta' not in v.name and 'gamma' not in v.name]

        # Loss
        one_hot_labels = slim.one_hot_encoding(labels_flat, params['num_classes'], 1.0, 0.0)
        loss_image = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_labels,
            logits=tf.reshape(logits, [-1, params['num_classes']])
        )
        loss_image = tf.reshape(loss_image, tf.shape(labels))
        label_weights, gaussian_edge = seg_modify_gradient_weight(
            labels, remain_rate=0.5, maximum=4.0)

        cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=one_hot_labels,
            logits=tf.reshape(logits, [-1, params['num_classes']]),
            weights=not_ignore_mask * tf.reshape(label_weights, [-1])
        )
        regularization_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)

        loss = cross_entropy + tf.add_n(regularization_losses)

        accuracy = tf.metrics.accuracy(
            labels_flat, preds_flat,
            weights=not_ignore_mask
        )
        mean_iou = tf.metrics.mean_iou(
            labels_flat, preds_flat, params['num_classes'],
            weights=not_ignore_mask
        )
        precision = tf.metrics.precision(
            labels_flat, preds_flat,
            weights=not_ignore_mask
        )
        recall = tf.metrics.recall(
            labels_flat, preds_flat,
            weights=not_ignore_mask
        )
        f1_max = tf.contrib.metrics.f1_score(
            labels_flat, preds_flat, num_thresholds=50,
            weights=not_ignore_mask
        )
        metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou, 'precision': precision, 'recall': recall,
                   'f1_max': f1_max}

        if mode == tf.estimator.ModeKeys.TRAIN:

            global_step = tf.train.get_or_create_global_step()

            learning_rate = tf.train.polynomial_decay(
                params['initial_learning_rate'],
                tf.cast(global_step, tf.int32) - params['initial_global_step'],
                params['max_iter'],
                params['end_learning_rate'],
                power=params['power']
            )

            with tf.name_scope('Summary'):
                with tf.name_scope('mean_iou'):
                    def compute_mean_iou_per_classes(total_cm, name='mean_iou'):
                        """Compute the mean intersection-over-union via the confusion matrix."""
                        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
                        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
                        cm_diag = tf.to_float(tf.diag_part(total_cm))
                        denominator = sum_over_row + sum_over_col - cm_diag

                        # The mean is only computed over classes that appear in the
                        # label or prediction tensor. If the denominator is 0, we need to
                        # ignore the class.
                        num_valid_entries = tf.reduce_sum(tf.cast(
                            tf.not_equal(denominator, 0), dtype=tf.float32))

                        # If the value of the denominator is 0, set it to 1 to avoid
                        # zero division.
                        denominator = tf.where(
                            tf.greater(denominator, 0),
                            denominator,
                            tf.ones_like(denominator))
                        iou = tf.div(cm_diag, denominator)

                        for i in range(params['num_classes']):
                            tf.identity(iou[i], name='train_iou_class{}'.format(i))
                            tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

                        # If the number of valid entries is 0 (no classes) we return 0.
                        result = tf.where(
                            tf.greater(num_valid_entries, 0),
                            tf.reduce_sum(iou, name=name) / num_valid_entries,
                            0)
                        return result

                    train_mean_iou = compute_mean_iou_per_classes(mean_iou[1])

                tf.summary.image(
                    'label_modified',
                    label_weights,
                    max_outputs=1
                )

                tf.summary.image(
                    'error_image',
                    loss_image,
                    max_outputs=1
                )

                # tf.summary.image(
                #     'confusion_weights',
                #     confusion_weights,
                #     max_outputs=1
                # )

                tf.summary.image(
                    'Result',
                    tf.concat([images, gt_decoded_labels, predict_decoded_labels], axis=2),
                    max_outputs=1
                )
                # Create a tensor for logging purposes.
                tf.identity(loss, name='loss')
                tf.summary.scalar('loss', loss)
                tf.identity(learning_rate, name='learning_rate')
                tf.summary.scalar('learning_rate', learning_rate)
                tf.identity(accuracy[1], name='train_px_accuracy')
                tf.summary.scalar('train_px_accuracy', accuracy[1])
                tf.identity(train_mean_iou, name='train_mean_iou')
                tf.summary.scalar('train_mean_iou', train_mean_iou)

            with tf.name_scope('Optimizer'):
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate,
                    momentum=params['momentum']
                )
                if FLAGS.num_gpu != 1:
                    optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

                # Batch norm requires update ops to be added as a dependency to the train_op
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(
                        loss=loss,
                        global_step=global_step,
                        var_list=train_var_list
                    )
        else:
            train_op = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics
        )
