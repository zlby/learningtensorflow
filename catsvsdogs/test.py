import tensorflow as tf
import numpy as np
import os
import matplotlib.pylab as plt


file = 'F:/Citibank/machine learning/TensorFlow/project/My-TensorFlow-tutorials/01 cats vs dogs/data/train/cat.0.jpg'


image_content = tf.read_file(file)
image = tf.image.decode_jpeg(image_content, channels=3)
print(image)
