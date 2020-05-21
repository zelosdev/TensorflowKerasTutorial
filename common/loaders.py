#####################
# importing modules #

import pickle
from glob import glob
import numpy as np
import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_mnist():

    # x_train: [60000, 28, 28] (28x28 grayscale images)
    # x_test: [10000, 28, 28] (28x28 grayscale images)
    # y_train: [60000, 1] (integer labels in the range 0 to 9)
    # y_test: [10000, 1] (integer labels in the range 0 to 9)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_train = x_train.reshape(x_train.shape + (1,))  # from (60000, 28, 28) to (60000, 28, 28, 1)

    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape + (1,))  # from (10000, 28, 28) to (10000, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)

def load_cifar10():

    # x_train: [50000, 32, 32, 3] (32x32 RGB images)
    # x_test: [10000, 32, 32, 3] (32x32 RGB images)
    # y_train: [50000, 1] (integer labels in the range 0 to 9)
    # y_test: [10000, 1] (integer labels in the range 0 to 9)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Neural networks work best when each input is inside the range -1 to 1.
    # Here, from 0~255 integer to 0.0~1.0 4bytes float
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # from integer labels to one-hot-encoded vectors
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def load_celeba_images(path, target_dim, batch_size):

    file_list = np.array(glob(os.path.join(path, '*/*.jpg')))
    num_images = len(file_list)

    data_gen = ImageDataGenerator(rescale=1./255.)

    data_flow = data_gen.flow_from_directory(
        directory   = path,
        target_size = target_dim[:2], # [:2]: except channels
        batch_size  = batch_size,
        shuffle     = True,
        class_mode  = 'input',
        subset      = 'training'
    )

    return num_images, data_flow

def load_quickdraw_camel():

    x = np.load('./data/full_numpy_bitmap_camel.npy')
    x = (x.astype('float32') - 127.5) / 127.5 # from [0, 255] to [-1.0, +1.0]
    x = x.reshape(x.shape[0], 28, 28, 1) # x.shape[0]: # of images

    return x # only images without lables

def load_model(model_class, path):

    with open(os.path.join(path, 'params.pkl'), 'rb') as f:
        arguments = pickle.load(f) # the arguments of the Autoencoder's constructor

    model = model_class(*arguments)
    model.load_weights(os.path.join(path, 'weights.h5'))

    return model