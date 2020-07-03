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

from common.Viewer import *

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

def load_quickdraw_camel():

    x = np.load('./data/full_numpy_bitmap_camel.npy')
    x = (x.astype('float32') - 127.5) / 127.5 # from [0, 255] to [-1.0, +1.0]
    x = x.reshape(x.shape[0], 28, 28, 1) # x.shape[0]: # of images

    return x # only images without lables

##########################
# test: loading an image #

x_train = load_quickdraw_camel()

print(x_train.shape) # (121399, 28, 28, 1)
plt.imshow(x_train[200,:,:,0], cmap = 'gray')
plt.show()

####################
# defining a class #

class GAN():

    def __init__(self):

        self.z_dim = 100

        self._build()

    def _build(self):

        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()

        input_layer = Input(shape=self.z_dim, name='model_input')
        output_layer = self.discriminator(self.generator(input_layer))
        self.combined_model = Model(input_layer, output_layer) # generator-discriminator combined

        self.discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0008))
        self.combined_model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0004))

    def _build_discriminator(self):

        input_layer = Input(shape=(28, 28, 1), name='d_input')
        x = input_layer

        x = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.4)(x)

        x = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.4)(x)

        x = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.4)(x)

        x = Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.4)(x)

        x = Flatten()(x)

        output_layer = Dense(1, activation='sigmoid')(x)

        return Model(input_layer, output_layer)

    def _build_generator(self):

        input_layer = Input(shape=self.z_dim)
        x = input_layer

        initial_dense_layer_size = (7, 7, 64)

        x = Dense(np.prod(initial_dense_layer_size))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)

        x = Reshape(initial_dense_layer_size)(x)

        x = UpSampling2D()(x)
        x = Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(filters=64, kernel_size=5, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(filters=64, kernel_size=5, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(filters=1, kernel_size=5, strides=1, padding='same')(x)
        x = Activation('tanh')(x)

        output_layer = x

        return Model(input_layer, output_layer)

    def train(self, x_train, batch_size, epochs):

        # adversarial loss ground truths (the answer for the decision)
        REAL = np.ones((batch_size, 1)) # the answer for the decision as real
        FAKE = np.zeros((batch_size, 1)) # the answer for the decision as fake

        for epoch in range(epochs):

            ####################################
            # training the discriminator model #
            self.discriminator.trainable = True

            # randomly selected images
            num_training_images = x_train.shape[0] # x_train.shape = (80000, 28, 28, 1)
            idx = np.random.randint(0, num_training_images, batch_size)
            real_imgs = x_train[idx] # how many images?: batch_size

            # randomly generated images
            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict(noise) # how many images?: batch_size

            loss_real = self.discriminator.train_on_batch(real_imgs, REAL)
            loss_fake = self.discriminator.train_on_batch(gen_imgs, FAKE)
            loss_D = 0.5 * (loss_real + loss_fake)

            ###############################
            # training the combined model #
            self.discriminator.trainable = False

            loss_G = self.combined_model.train_on_batch(noise, REAL)

            print("%d [D loss: (%.3f)(R %.3f, F %.3f)]  [G loss: %.3f] " % (epoch, loss_D, loss_real, loss_fake, loss_G))

            if epoch % 100 == 0:
                self._export_result(epoch)

    def _export_result(self, epoch):

        viewer = Viewer(num_rows=5, num_cols=5, width=15, height=15)

        for i in range(5):
            noise = np.random.normal(0, 1, (5, self.z_dim))
            imgs = self.generator.predict(noise)
            imgs = ((127.5 * imgs) + 127.5).astype('uint8').clip(0, 255) # from [-1.0, +1.0] to [0, 255]
            viewer.add_row(imgs)

        viewer.save('result/QuickDraw_GAN/sample_%d.png' % epoch)

gan = GAN()
gan.train(x_train, batch_size=256, epochs=6000)