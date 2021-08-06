import numpy as np
import tensorflow as tf
from skimage import io
from matplotlib import pyplot as plt

mask_2 = r'image/mask_single_2.tif'
mask = io.imread(mask_2).astype(np.float32)
inputs = np.expand_dims(mask, axis=(0, -1))

# Laplacian for edge extraction
laplacian_filter = tf.constant([[0, .25, 0], [.25, -1, .25], [0, .25, 0]],
                               dtype=tf.float32)
laplacian_filter = tf.reshape(laplacian_filter, (3, 3, 1, 1))

output = tf.nn.conv2d(inputs, filters=laplacian_filter, strides=1, padding='SAME')

edge = output != 0
edge = tf.cast(edge, tf.float32)

# Gaussian blur for generating pixel weight


def gaussian_2d(ksize, sigma=1):
    m = (ksize - 1) / 2
    y, x = np.ogrid[-m:m+1, -m:m+1]
    value = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    # value[value < np.finfo(value.dtype).eps * value.max()] = 0
    sum_v = value.sum()
    if sum_v != 0:
        value /= sum_v
    return value


gaussian_filter = gaussian_2d(ksize=3, sigma=1)
gaussian_filter = np.reshape(gaussian_filter, (3, 3, 1, 1))

weight = 10 * tf.nn.conv2d(edge, filters=gaussian_filter, strides=1, padding='SAME') + 1
print(weight.shape)
# print(gaussian_filter)
#
#
io.imshow(weight.numpy()[0, :, :, 0], cmap=plt.cm.inferno)
plt.show()



