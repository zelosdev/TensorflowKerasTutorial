#####################
# importing modules #

from common.utils import *
from common.loaders import *
from common.Viewer import *

from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

################
# loading data #

(x_train, y_train), (x_test, y_test) = load_cifar10()

######################
# building the model #

class FCN():

    def __init__(self, method=1):
        self.method = method
        self._build()

    def _build(self):

        if self.method == 1:

            self.model = Sequential([
                Flatten(input_shape=(32, 32, 3)),
                Dense(200, activation='relu'),
                Dense(150, activation='relu'),
                Dense(10, activation='softmax')  # 10: the number of classes
            ])

        elif self.method == 2:

            self.model = Sequential()
            self.model.add(Flatten(input_shape=(32, 32, 3)))
            self.model.add(Dense(200, activation='relu'))
            self.model.add(Dense(150, activation='relu'))
            self.model.add(Dense(10, activation='softmax'))

        elif self.method == 3:

            input_layer = Input(shape=(32, 32, 3))
            x = input_layer
            x = Flatten()(x)
            x = Dense(200, activation='relu')(x)
            x = Dense(150, activation='relu')(x)
            output_layer = Dense(10, activation='softmax')(x)
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

fcn = FCN(method=1)

#######################
# compiling the model #

fcn.compile(learning_rate=0.0005)

######################
# training the model #

fcn.train(x_train, y_train, batch_size=32, epochs=10)

########################
# evaluating the model #

fcn.evaluate(x_test, y_test, batch_size=1000)

######################
# drawing the result #

indices = select_indices(len(x_test), 10)
x = select_items(x_test, indices)
y = select_items(y_test, indices)

answer = to_classes(y, 'CIFAR10')
predicted = to_classes(fcn.predict(x), 'CIFAR10')

viewer = Viewer(num_rows=2, num_cols=10, width=15, height=2)
viewer.add_row(x, answer, predicted)
viewer.show()