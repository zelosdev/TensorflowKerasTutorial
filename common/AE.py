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

import pickle
import numpy as np

from common.utils import create_directory
from common.callbacks import OutputAEImgCallback

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

######################
# defining the class #

class AE():

    def __init__(self,
        input_dim, z_dim,
        encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides, # parameters for encoder
        decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides, # parameters for decoder
        use_batch_norm, use_dropout, # parameters for regularization
        export_path
    ):

        self.input_dim                  = input_dim
        self.z_dim                      = z_dim
        self.encoder_conv_filters       = encoder_conv_filters
        self.encoder_conv_kernel_size   = encoder_conv_kernel_size
        self.encoder_conv_strides       = encoder_conv_strides
        self.decoder_conv_t_filters     = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides     = decoder_conv_t_strides
        self.use_batch_norm             = use_batch_norm
        self.use_dropout                = use_dropout
        self.export_path                = export_path

        self._build()

    def _build(self):

        input_dim                  = self.input_dim
        z_dim                      = self.z_dim
        encoder_conv_filters       = self.encoder_conv_filters
        encoder_conv_kernel_size   = self.encoder_conv_kernel_size
        encoder_conv_strides       = self.encoder_conv_strides
        decoder_conv_t_filters     = self.decoder_conv_t_filters
        decoder_conv_t_kernel_size = self.decoder_conv_t_kernel_size
        decoder_conv_t_strides     = self.decoder_conv_t_strides
        use_batch_norm             = self.use_batch_norm
        use_dropout                = self.use_dropout

        ### encoder

        num_encoder_layers = len(encoder_conv_filters)

        encoder_input_layer = Input(shape=input_dim, name='encoder_input')
        x = encoder_input_layer

        for i in range(num_encoder_layers):
            x = Conv2D(
                filters     = encoder_conv_filters[i],
                kernel_size = encoder_conv_kernel_size[i],
                strides     = encoder_conv_strides[i],
                padding     = 'same',
                name        = 'encoder_conv' + str(i)
            )(x)

            x = LeakyReLU()(x)

            if use_batch_norm:
                x = BatchNormalization()(x)
            if use_dropout:
                x = Dropout(rate=0.3)(x)

        shape_before_flattening = x.shape[1:]  # 1: from (None, 7, 7, 64) to (7, 7, 64)

        x = Flatten()(x)

        encoder_output_layer = Dense(z_dim, name='encoder_output')(x) # the latent space
        self.encoder = Model(encoder_input_layer, encoder_output_layer, name='Encoder')

        ### decoder

        num_decoder_layers = len(decoder_conv_t_filters)

        decoder_input_layer = Input(shape=z_dim, name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input_layer)
        x = Reshape(shape_before_flattening)(x)

        for i in range(num_decoder_layers):
            x = Conv2DTranspose(
                filters       = decoder_conv_t_filters[i],
                kernel_size = decoder_conv_t_kernel_size[i],
                strides     = decoder_conv_t_strides[i],
                padding     = 'same',
                name        = 'decoder_conv_t' + str(i)
            )(x)

            if i < (num_decoder_layers-1):
                x = LeakyReLU()(x)

                if use_batch_norm:
                    x = BatchNormalization()(x)
                if use_dropout:
                    x = Dropout(rate=0.3)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output_layer = x
        self.decoder = Model(decoder_input_layer, decoder_output_layer, name='Decoder')

        model_input_layer = encoder_input_layer
        model_output_layer = self.decoder(encoder_output_layer)

        self.model = Model(model_input_layer, model_output_layer, name='Autoencoder')

        self._save()

    def compile(self, learning_rate):

        opt = Adam(lr=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    def train(self, x_train, batch_size, epochs, with_generator=False):

        export_path     = self.export_path
        steps_per_epoch = len(x_train) / batch_size

        # callbacks
        save_weights_callback = ModelCheckpoint(os.path.join(export_path, 'weights.h5'), save_weights_only=True, save_freq='epoch', verbose=1)
        output_image_callback = OutputAEImgCallback(export_path, 100, self)
        callback_list = [save_weights_callback, output_image_callback]

        if with_generator:
            self.model.fit_generator(x_train, steps_per_epoch=steps_per_epoch, epochs=epochs, shuffle=True, verbose=1, callbacks=callback_list)
        else:
            self.model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, callbacks=callback_list)

    def _save(self):

        create_directory(self.export_path)

        # The number and order of constructor arguments must match.
        with open(os.path.join(self.export_path, 'params.pkl'), 'wb') as f:
            pickle.dump([ # the arguments of the class constructor
                self.input_dim, self.z_dim,
                self.encoder_conv_filters, self.encoder_conv_kernel_size, self.encoder_conv_strides,
                self.decoder_conv_t_filters, self.decoder_conv_t_kernel_size, self.decoder_conv_t_strides,
                self.use_batch_norm, self.use_dropout,
                self.export_path
            ], f)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)