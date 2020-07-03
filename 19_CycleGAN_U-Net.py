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
from glob import glob

from PIL import Image
import imageio
from collections import deque
import random
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, LeakyReLU, Activation, Concatenate
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose

class DataLoader():
    def __init__(self, img_size):
        self.img_size = img_size

    def load_random_images(self, domain='A', how_many=1, is_testing=False):
        directory_name = 'train%s' % domain if not is_testing else 'test%s' % domain # trainA, trainB, testA, testB
        all_file_paths = glob('./data/apple_and_orange/%s/*' % directory_name) # all file paths
        selected_paths = np.random.choice(all_file_paths, size=how_many) # randomly select N images

        imgs = []

        for path in selected_paths:
            img = self._imread(path) # load
            img = np.array(Image.fromarray(img).resize(self.img_size)) # resize

            if not is_testing:
                if np.random.random() > 0.5:
                    img = np.fliplr(img) # flip left to right

            imgs.append(img) # append

        imgs = np.array(imgs)/127.5 - 1.0 # pixel value range: -1.0 ~ +1.0

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = 'train' if not is_testing else 'test'
        all_file_paths_A = glob('./data/apple_and_orange/%sA/*' % data_type) # all file paths
        all_file_paths_B = glob('./data/apple_and_orange/%sB/*' % data_type) # all file paths

        # len(all_file_paths_A): 995
        # len(all_file_paths_b): 1019
        self.num_batches = int(min(len(all_file_paths_A), len(all_file_paths_B)) / batch_size)
        total_samples = self.num_batches * batch_size

        # Why are we doing this? This is because the number of two sets of images is different.
        all_file_paths_A = np.random.choice(all_file_paths_A, total_samples, replace=False)
        all_file_paths_B = np.random.choice(all_file_paths_B, total_samples, replace=False)

        for i in range(self.num_batches-1):
            batch_A = all_file_paths_A[i*batch_size:(i+1)*batch_size]
            batch_B = all_file_paths_B[i*batch_size:(i+1)*batch_size]

            imgs_A, imgs_B = [], []

            for img_A, img_B in zip(batch_A, batch_B):

                # load
                img_A = self._imread(img_A)
                img_B = self._imread(img_B)

                # resize
                img_A = np.array(Image.fromarray(img_A).resize(self.img_size))
                img_B = np.array(Image.fromarray(img_B).resize(self.img_size))

                # flip left to right
                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                # append
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            # pixel value range: -1.0 ~ +1.0
            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self._imread(path)
        img = np.array(Image.fromarray(img).resize(self.img_size))
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def _imread(self, path):
        return imageio.imread(path, pilmode='RGB').astype(np.uint8)


class CycleGAN():

    def __init__(self, input_dim, learning_rate):
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # tensor size
        self.img_width = input_dim[0] # rows
        self.img_height = input_dim[1] # columns
        self.img_channels = input_dim[2] # depth
        self.img_shape = (self.img_width, self.img_height, self.img_channels)

        self.epoch = 0
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_width / 2**3) # 128 / 8
        self.disc_patch = (patch, patch, 1) # (16, 16, 1)

        self._build()

    def _build(self):

        self.d_A = self._build_discriminator()
        self.d_B = self._build_discriminator()
        
        self.d_A.compile(loss='mse', optimizer=Adam(self.learning_rate, 0.5), metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=Adam(self.learning_rate, 0.5), metrics=['accuracy'])

        self.g_AB = self._build_generator()
        self.g_BA = self._build_generator()

        # for the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # original images (input images from both domains)
        O_A = Input(shape=self.img_shape, name='input_image_A')
        O_B = Input(shape=self.img_shape, name='input_image_B')

        # fake images (generated images to the other domain)
        F_A = self.g_BA(O_B)
        F_B = self.g_AB(O_A)

        # reconstructed images (generated images to the original domain)
        R_A = self.g_BA(F_B)
        R_B = self.g_AB(F_A)

        # identity images (generated images to the same domain)
        I_A = self.g_BA(O_A)
        I_B = self.g_AB(O_B)

        # validity to be determined by the discriminators
        V_A = self.d_A(F_A)
        V_B = self.d_B(F_B)

        # combined model to train the generates to fool discriminators
        self.combined_model = Model(inputs=[O_A, O_B], outputs=[V_A, V_B, R_A, R_B, I_A, I_B])
        self.combined_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], loss_weights=[1, 1, 10, 10, 2, 2], optimizer=Adam(self.learning_rate, 0.5))

        self.d_A.trainable = True
        self.d_B.trainable = True

    def _build_generator(self):

        input_layer = Input(shape=self.img_shape)
        x = input_layer

        # Downsampling

        x = Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = Activation('relu')(x)
        d1 = x

        x = Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = Activation('relu')(x)
        d2 = x

        x = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = Activation('relu')(x)
        d3 = x

        x = Conv2D(filters=256, kernel_size=4, strides=2, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = Activation('relu')(x)
        d4 = x

        # Upsampling

        x = UpSampling2D(size=2)(d4)
        x = Conv2D(filters=128, kernel_size=4, strides=1, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = Activation('relu')(x)
        u1 = Concatenate()([x, d3])

        x = UpSampling2D(size=2)(u1)
        x = Conv2D(filters=64, kernel_size=4, strides=1, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = Activation('relu')(x)
        u2 = Concatenate()([x, d2])

        x = UpSampling2D(size=2)(u2)
        x = Conv2D(filters=32, kernel_size=4, strides=1, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = Activation('relu')(x)
        u3 = Concatenate()([x, d1])

        # upsampling to return the tensor to the same size as the original image
        x = UpSampling2D(size=2)(u3)
        output_layer = Conv2D(self.img_channels, kernel_size=4, strides=1, padding='same', activation='tanh')(x)

        return Model(input_layer, output_layer)

    def _build_discriminator(self):

        input_layer = Input(shape=self.img_shape)
        x = input_layer

        # A CycleGAN discriminator is a series of convolutional layers, all with instance normalization (except the first layer).
        x = Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=256, kernel_size=4, strides=1, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = LeakyReLU(0.2)(x)

        # The final layer is a convolutional layer with only one filter and no activation.
        output_layer = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)

        return Model(input_layer, output_layer)

    def train(self, data_loader, run_folder, epochs, test_A_file, test_B_file, batch_size=1, sample_interval=50):

        # adversarial loss ground truths (the answer for the decision)
        REAL = np.ones((batch_size,) + self.disc_patch) # (1, 16, 16, 1) = (1,,) + (16, 16, 1)
        FAKE = np.zeros((batch_size,) + self.disc_patch) # (1, 16, 16, 1) = (1,,) + (16, 16, 1)

        for epoch in range(self.epoch, epochs):
            for batch, (imgs_A, imgs_B) in enumerate(data_loader.load_batch()): # batch_size = 1

                d_loss, d_acc = self._train_discriminators(imgs_A, imgs_B, REAL, FAKE)
                g_loss = self._train_combined_model(imgs_A, imgs_B, REAL)

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] " \
                    % ( self.epoch, epochs,
                        batch, data_loader.num_batches,
                        d_loss, 100*d_acc,
                        g_loss[0],
                        np.sum(g_loss[1:3]), # loss for adversarial term
                        np.sum(g_loss[3:5]), # loss for reconstruction term
                        np.sum(g_loss[5:7]))) # loss for identity term

                if batch % sample_interval == 0:
                    self._export_result(data_loader, batch, run_folder, test_A_file, test_B_file)

            self.epoch += 1

    def _train_discriminators(self, O_A, O_B, REAL, FAKE):

        F_A = self.g_BA.predict(O_B)
        F_B = self.g_AB.predict(O_A)

        dA_ret_real = self.d_A.train_on_batch(O_A, REAL)
        dA_ret_fake = self.d_A.train_on_batch(F_A, FAKE)
        dA_ret = 0.5 * np.add(dA_ret_real, dA_ret_fake)
        dA_loss, dA_acc = dA_ret[0], dA_ret[1]

        dB_ret_real = self.d_B.train_on_batch(O_B, REAL)
        dB_ret_fake = self.d_B.train_on_batch(F_B, FAKE)
        dB_ret = 0.5 * np.add(dB_ret_real, dB_ret_fake)
        dB_loss, dB_acc = dB_ret[0], dB_ret[1]

        d_loss_total = 0.5 * np.add(dA_loss, dB_loss)
        d_acc_total = 0.5 * np.add(dA_acc, dB_acc)

        return (d_loss_total, d_acc_total)

    def _train_combined_model(self, imgs_A, imgs_B, real):

        return self.combined_model.train_on_batch([imgs_A, imgs_B], [real, real, imgs_A, imgs_B, imgs_A, imgs_B])

    def _export_result(self, data_loader, i_batch, run_folder, test_A_file, test_B_file):

        for p in range(2): # to use the specified image and the random image alternately

            if p == 0: # using specified images
                O_A = data_loader.load_img('data/apple_and_orange/testA/%s' % test_A_file)
                O_B = data_loader.load_img('data/apple_and_orange/testB/%s' % test_B_file)
            else: # using randomly selected images
                O_A = data_loader.load_random_images(domain='A', how_many=1, is_testing=True)
                O_B = data_loader.load_random_images(domain='B', how_many=1, is_testing=True)

            F_A = self.g_BA.predict(O_B)
            F_B = self.g_AB.predict(O_A)

            R_A = self.g_BA.predict(F_B)
            R_B = self.g_AB.predict(F_A)

            I_A = self.g_BA.predict(O_A)
            I_B = self.g_AB.predict(O_B)

            imgs = np.concatenate([O_A, F_B, R_A, I_A, O_B, F_A, R_B, I_B])
            imgs = 0.5 * imgs + 0.5
            imgs = np.clip(imgs, 0, 1)

            r, c = 2, 4 # rows, columns
            idx = 0 # index: 0, 1, 2, 3

            titles = ['Original', 'Translated', 'Reconstructed', 'ID']
            fig, axs = plt.subplots(r, c, figsize=(25, 12.5))

            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(imgs[idx])
                    axs[i,j].set_title(titles[j])
                    axs[i,j].axis('off')
                    idx += 1

            fig.savefig(os.path.join(run_folder, '%d_%d_%d.png' % (p, self.epoch, i_batch)))
            plt.close()


IMAGE_SIZE = 128

data_loader = DataLoader(img_size=(IMAGE_SIZE, IMAGE_SIZE))

gan = CycleGAN(input_dim=(IMAGE_SIZE, IMAGE_SIZE, 3), learning_rate=0.0002)

BATCH_SIZE = 1
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 10

TEST_A_FILE = 'n07740461_14740.jpg'
TEST_B_FILE = 'n07749192_4241.jpg'

gan.train(data_loader
        , run_folder='C:/work/KerasTest/result/Cycle_U-Net'
        , epochs=EPOCHS
        , test_A_file=TEST_A_FILE
        , test_B_file=TEST_B_FILE
        , batch_size=BATCH_SIZE
        , sample_interval=PRINT_EVERY_N_BATCHES)