##########################################################
# to turn off warnings                                   #
import warnings                                          #
warnings.filterwarnings('ignore')                        #
                                                         #
import os                                                #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                 #
                                                         #
import tensorflow.python.util.deprecation as deprecation #
deprecation._PRINT_DEPRECATION_WARNINGS = False          #
                                                         #
##########################################################

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow import keras

img_path = 'data/image/tiger.jpg'

img0 = keras.preprocessing.image.load_img(img_path)
width, height = img0.size # 1600 1200

# from 'PIL.JpegImagePlugin.JpegImageFile' to 'numpy.ndarray'
img0 = keras.preprocessing.image.img_to_array(img0)
print(img0.shape) # (1200, 1600, 3): (height, width, channels)
print(img0.min(), ' ~ ', img0.max()) # 0.0 ~ 255.0

img1 = img0[::-1, :, :] # flipping vertically
img2 = img0[:, ::-1, :] # flipping horizontally

# from 'numpy.ndarray' to 'tensorflow.python.framework.ops.EagerTensor'
img0 = tf.convert_to_tensor(img0)
img1 = tf.convert_to_tensor(img1)
img2 = tf.convert_to_tensor(img2)

def show_image(img0, img1, img2, axis):
    img = tf.concat([img0, img1, img2], axis=0)

    # from 'tensorflow.python.framework.ops.EagerTensor' to 'numpy.ndarray'
    img = img.numpy()
    img = np.clip(img, 0, 255).astype('uint8')

    plt.imshow(img)
    plt.show()

show_image(img0, img1, img2, 0)
show_image(img0, img1, img2, 1)
