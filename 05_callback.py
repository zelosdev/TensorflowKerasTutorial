#####################
# importing modules #

from common.utils import *
from common.loaders import *

from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

################
# loading data #

(x_train, y_train), (x_test, y_test) = load_cifar10()

######################
# building the model #

model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(200, activation='relu'),
    Dense(150, activation='relu'),
    Dense(10, activation='softmax')
])

#######################
# compiling the model #

opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#####################
# defining callback #

import tensorflow as tf

class MyCallback1(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        print('Training step is started.')
    def on_train_end(self, logs={}):
        print('Training step is finished.')

class MyCallback2(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1
        print('Epoch', self.epoch, 'begin')
    def on_epoch_end(self, epoch, logs={}):
        print('Epoch', self.epoch, 'end')
        print('Accuracy:', logs.get('accuracy'))
        if logs.get('accuracy') > 0.4:
            self.model.stop_training = True

cb1 = MyCallback1()
cb2 = MyCallback2()

######################
# training the model #

# verbose
# 0: silent
# 1: progress bar
# 2: one line per epoch
model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True, verbose=0, callbacks=[cb1, cb2])