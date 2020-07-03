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

import numpy as np
import matplotlib.pyplot as plt

from common.Viewer import *

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

class GAN():

    def __init__(self,
        input_dim, z_dim,
        d_conv_filters, d_conv_kernel_size, d_conv_strides, d_batch_norm_momentum, d_dropout_rate, # parameters for discriminator
        g_initial_dense_layer_size, g_upsample,                                                    # parameters for generator (upsampling)
        g_conv_filters, g_conv_kernel_size, g_conv_strides, g_batch_norm_momentum, g_dropout_rate, # parameters for generator (CNN)
    ):

        self.input_dim                  = input_dim
        self.z_dim                      = z_dim

        self.d_conv_filters             = d_conv_filters
        self.d_conv_kernel_size         = d_conv_kernel_size
        self.d_conv_strides             = d_conv_strides
        self.d_batch_norm_momentum      = d_batch_norm_momentum
        self.d_dropout_rate             = d_dropout_rate

        self.g_initial_dense_layer_size = g_initial_dense_layer_size
        self.g_upsample                 = g_upsample

        self.g_conv_filters             = g_conv_filters
        self.g_conv_kernel_size         = g_conv_kernel_size
        self.g_conv_strides             = g_conv_strides
        self.g_batch_norm_momentum      = g_batch_norm_momentum
        self.g_dropout_rate             = g_dropout_rate

        self._build()

    def _build(self):

        self._build_discriminator()
        self._build_generator()

        input_layer = Input(shape=self.z_dim, name='model_input')
        output_layer = self.discriminator(self.generator(input_layer))
        self.combined_model = Model(input_layer, output_layer)

    def _build_discriminator(self):

        num_layers = len(self.d_conv_filters)

        input_layer = Input(shape=self.input_dim, name='d_input')
        x = input_layer

        for i in range(num_layers):

            x = Conv2D(
                filters     = self.d_conv_filters[i],
                kernel_size = self.d_conv_kernel_size[i],
                strides     = self.d_conv_strides[i],
                padding     = 'same',
                name        = 'd_conv_' + str(i)
            )(x)

            if self.d_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum=self.d_batch_norm_momentum)(x)

            x = LeakyReLU()(x)

            if self.d_dropout_rate:
                x = Dropout(rate=self.d_dropout_rate)(x)

        x = Flatten()(x)

        output_layer = Dense(1, activation='sigmoid')(x)

        self.discriminator = Model(input_layer, output_layer)

    def _build_generator(self):

        num_layers = len(self.g_conv_filters)

        input_layer = Input(shape=self.z_dim, name='g_input')
        x = input_layer

        x = Dense(np.prod(self.g_initial_dense_layer_size))(x)

        if self.g_batch_norm_momentum:
            x = BatchNormalization(momentum=self.g_batch_norm_momentum)(x)

        x = LeakyReLU()(x)

        x = Reshape(self.g_initial_dense_layer_size)(x)

        if self.g_dropout_rate:
            x = Dropout(rate=self.g_dropout_rate)(x)

        for i in range(num_layers):

            if self.g_upsample[i] == 2:

                x = UpSampling2D()(x)
                x = Conv2D(
                    filters     = self.g_conv_filters[i],
                    kernel_size = self.g_conv_kernel_size[i],
                    padding     = 'same',
                    name        = 'g_conv_' + str(i)
                )(x)

            else:

                x = Conv2DTranspose(
                    filters     = self.g_conv_filters[i],
                    kernel_size = self.g_conv_kernel_size[i],
                    padding     = 'same',
                    strides     = self.g_conv_strides[i],
                    name        = 'g_conv_' + str(i)
                )(x)

            if i < num_layers - 1:

                if self.g_batch_norm_momentum:
                    x = BatchNormalization(momentum=self.g_batch_norm_momentum)(x)

                x = LeakyReLU()(x)

            else:

                x = Activation('tanh')(x)

        output_layer = x

        self.generator = Model(input_layer, output_layer)

    def compile(self, d_learning_rate, g_learning_rate):

        self.discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=d_learning_rate))
        self.combined_model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=g_learning_rate))

    def _train_D(self, x_train, batch_size):

        self.discriminator.trainable = True

        real = np.ones((batch_size, 1)) # the answer for the decision as real
        fake = np.zeros((batch_size, 1)) # the answer for the decision as fake

        num_training_images = x_train.shape[0] # x_train.shape = (80000, 28, 28, 1)
        idx = np.random.randint(0, num_training_images, batch_size)
        true_imgs = x_train[idx] # how many images?: batch_size

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise) # how many images?: batch_size

        loss_real = self.discriminator.train_on_batch(true_imgs, real)
        loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        loss_D = 0.5 * (loss_real + loss_fake) # [0]: loss, [1]: accuracy

        return [loss_D, loss_real, loss_fake]

    def _train_G(self, batch_size):

        self.discriminator.trainable = False

        real = np.ones((batch_size, 1)) # the answer for the decision as real
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))

        loss_G = self.combined_model.train_on_batch(noise, real)

        return loss_G # [0]: loss, [1]: accuracy

    def train(self, x_train, batch_size, epochs):

        for epoch in range(epochs):

            loss_D = self._train_D(x_train, batch_size)
            loss_G = self._train_G(batch_size)

            print("%d [D loss: (%.3f)(R %.3f, F %.3f)]  [G loss: %.3f] " % (epoch, loss_D[0], loss_D[1], loss_D[2], loss_G))

            if epoch % 100 == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):

        viewer = Viewer(num_rows=5, num_cols=5, width=15, height=15)

        for i in range(5):
            noise = np.random.normal(0, 1, (5, self.z_dim))
            imgs = self.generator.predict(noise)
            imgs = ((127.5 * imgs) + 127.5).astype('uint8').clip(0, 255) # from [-1.0, +1.0] to [0, 255]
            viewer.add_row(imgs)

        viewer.save('result/QuickDraw_GAN/sample_%d.png' % epoch)