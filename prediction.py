import os
import sys
import cv2 as cv
from utils.preprocessing import *
from network.lane_segmentator import Segmentator
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from utils.base_util import Timer, file_list

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_gpu', 1,
                     'Number of GPUs to use.')

model_dir = './model'
image_size = (1080, 720)
data_dir = 'D:/HighwayDriving/Test/'

folder_list = file_list(data_dir)
images = list()
print('[Read images]')
for f in folder_list:
    image_files = file_list(os.path.join(f, 'image'))
    for i, i_f in enumerate(image_files):
        if i == 30:
            break
        img = np.array(Image.open(i_f).resize(image_size, Image.ANTIALIAS)).astype(np.float32)
        images.append(img)

num_images = len(images)
print('[Done]')

sess = tf.Session()
input_image = tf.placeholder(dtype=tf.float32, shape=[None, image_size[1], image_size[0], 3])

segmentation = Segmentator(
    params={
        'base_architecture': 'resnet_v2_50',
        'batch_size': 1,
        'fine_tune_batch_norm': False,
        'num_classes': 2,
        'weight_decay': 0.0001,
        'output_stride': 16,
        'batch_norm_decay': 0.9997
    }
)

logits = segmentation.network(inputs=input_image, is_training=False)

predict_classes = tf.expand_dims(
    tf.argmax(logits, axis=3, output_type=tf.int32),
    axis=3
)

variables_to_restore = tf.contrib.slim.get_variables_to_restore()
get_ckpt = tf.train.init_from_checkpoint(
    ckpt_dir_or_file='./model',
    assignment_map={v.name.split(':')[0]: v for v in variables_to_restore}
)

sess.run(tf.global_variables_initializer())


print('[Do segment]')
timer = Timer(as_progress_notifier=True)


def print_function():
    sys.stdout.write('\rLoad images : [%d/%d][%.2f%%][%d hour %d minute %d second left]' %
                     (timer.current_progress, timer.whole_number_of_data,
                      timer.current_progress / timer.whole_number_of_data * 100,
                      int(timer.remain_time / 3600), int(timer.remain_time / 60) % 60, timer.remain_time % 60))
    sys.stdout.flush()


timer.print_fn = print_function


video_codec = cv.VideoWriter_fourcc(*'XVID')
original_video_out = cv.VideoWriter('seq_original.avi', video_codec, 15, image_size)
overlay_video_out = cv.VideoWriter('seq_overlay.avi', video_codec, 15, image_size)
seg_video_out = cv.VideoWriter('seq_seg.avi', video_codec, 15, image_size)

timer.start(len(images))
for i in images:
    image_data = i
    predictions = sess.run(
        predict_classes,
        feed_dict={
            input_image: np.expand_dims(image_data, 0)
        }
    )
    red_image = np.transpose(np.tile(predictions[0], [1, 1, 3]), [2, 0, 1])
    red_image[0] = red_image[0] * 255
    red_image[1] = red_image[1] * 0
    red_image[2] = red_image[2] * 0
    red_image = np.transpose(red_image, [1, 2, 0])
    overlay = (red_image * 0.4 + i * 0.6).astype(np.uint8)
    original_video_out.write(cv.cvtColor(i.astype(np.uint8), cv.COLOR_RGB2BGR))
    overlay_video_out.write(cv.cvtColor(overlay, cv.COLOR_RGB2BGR))
    seg_video_out.write(cv.cvtColor(red_image.astype(np.uint8), cv.COLOR_RGB2BGR))
    timer.tick_timer()

original_video_out.release()
overlay_video_out.release()
seg_video_out.release()
