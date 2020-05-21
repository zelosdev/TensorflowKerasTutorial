#####################
# importing modules #

from common.utils import *
from common.loaders import *
from common.Viewer import *

from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

################
# loading data #

(x_train, y_train), (x_test, y_test) = load_cifar10()

######################
# building the model #

class CNN():

    def __init__(self, input_dim, conv_filters, conv_kernel_size, conv_strides, use_batch_norm, use_dropout):

        self.input_dim        = input_dim
        self.conv_filters     = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides     = conv_strides
        self.use_batch_norm   = use_batch_norm
        self.use_dropout      = use_dropout

        self._build()

    def _build(self):

        num_layers = len(self.conv_filters)

        input_layer = Input(shape=self.input_dim, name='input_layer')
        x = input_layer

        for i in range(num_layers):

            x = Conv2D(
                filters     = self.conv_filters[i],
                kernel_size = self.conv_kernel_size[i],
                strides     = self.conv_strides[i],
                padding     = 'same',
                name        = 'conv_layer' + str(i)
            )(x)

            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)
            if self.use_dropout:
                x = Dropout(rate=0.3)(x)

        x = Flatten()(x)

        x = Dense(128)(x)
        x = LeakyReLU()(x)

        if self.use_batch_norm:
            x = BatchNormalization()(x)
        if self.use_dropout:
            x = Dropout(rate=0.3)(x)

        output_layer = Dense(units=10, activation='softmax')(x)

        self.model = Model(input_layer, output_layer)

        show_structure(self.model, __file__)

    def compile(self, learning_rate):
        opt = Adam(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def train(self, x_train, y_train, batch_size, epochs):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    def evaluate(self, x_test, y_test, batch_size):
        self.model.evaluate(x_test, y_test, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

cnn = CNN(
    input_dim        = (32, 32, 3),
    conv_filters     = [32, 32, 64, 64],
    conv_kernel_size = [ 3,  3,  3,  3],
    conv_strides     = [ 1,  2,  1,  2],
    use_batch_norm   = False,
    use_dropout      = False
)

#######################
# compiling the model #

cnn.compile(learning_rate=0.0005)

######################
# training the model #

cnn.train(x_train, y_train, batch_size=32, epochs=10)

########################
# evaluating the model #

cnn.evaluate(x_test, y_test, batch_size=1000)

######################
# drawing the result #

indices = select_indices(len(x_test), 10)
x = select_items(x_test, indices)
y = select_items(y_test, indices)

answer = to_classes(y, 'CIFAR10')
predicted = to_classes(cnn.predict(x), 'CIFAR10')

viewer = Viewer(num_rows=2, num_cols=10, width=15, height=2)
viewer.add_row(x, answer, predicted)
viewer.show()