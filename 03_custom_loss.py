#####################
# importing modules #

from common.utils import *
from common.loaders import *
from common.Viewer import *

from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

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

method = 'CEE'

def my_loss(y_true, y_pred):
    if method == 'MSE': # mean square error
        return K.mean(K.square(y_true - y_pred))
    elif method == 'CEE': # cross entropy error
        return -K.sum(y_true * K.log(y_pred))

opt = Adam(lr=0.0005)
model.compile(loss=my_loss, optimizer=opt, metrics=['accuracy'])

######################
# training the model #

model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

########################
# evaluating the model #

model.evaluate(x_test, y_test, batch_size=1000)

######################
# drawing the result #

indices = select_indices(len(x_test), 10)
x = select_items(x_test, indices)
y = select_items(y_test, indices)

answer = to_classes(y, 'CIFAR10')
predicted = to_classes(model.predict(x), 'CIFAR10')

viewer = Viewer(num_rows=2, num_cols=10, width=15, height=2)
viewer.add_row(x, answer, predicted)
viewer.show()