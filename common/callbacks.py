############################
# turning off the warnings #

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

#####################
# importing modules #

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback

class OutputAEImgCallback(Callback):

    def __init__(self, export_path, every_n_batches, ae):
        self.epoch = 0
        self.batch = 0
        self.export_path = export_path
        self.every_n_batches = every_n_batches
        self.ae = ae

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1

    def on_batch_begin(self, batch, logs={}):
        self.batch += 1

    def on_batch_end(self, batch, logs={}):
        if batch % self.every_n_batches == 0:
            z_new = np.random.normal(size=(1, self.ae.z_dim))

            # squeeze: for example, from (28, 28, 1) to (28, 28) for MNIST data
            output_image = self.ae.decoder.predict(np.array(z_new))[0].squeeze()

            filepath = os.path.join(self.export_path, 'output_image_' + str(self.epoch).zfill(4) + '_' + str(self.batch).zfill(4) + '.jpg')

            if len(output_image.shape) == 2:
                plt.imsave(filepath, output_image, cmap='gray_r')
            else:
                plt.imsave(filepath, output_image)