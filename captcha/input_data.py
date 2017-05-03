import tensorflow as tf
import numpy as np
import os


def get_files(file_dir):
    images = []
    labels = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        images.append(file_dir + file)
        labels.append(name[0])

    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    #label = tf.cast(label, tf.string)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    #image = tf.image.rgb_to_grayscale(image)

    image_batch, label_batch = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=10, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    #label_batch = tf.cast(label_batch, tf.string)

    return image_batch, label_batch

'''
import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 10
IMG_W = 60
IMG_H = 160

train_dir = 'test/'

image_list, label_list = get_files(train_dir)

image_batch, label_batch = get_batch(
    image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)


with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i < 1:

            img, label = sess.run([image_batch, label_batch])

            # just test one batch
            for j in np.arange(BATCH_SIZE):
                print('label: %s' % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''