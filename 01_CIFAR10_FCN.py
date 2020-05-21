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

# how to get a pixel value
# 1: green channel, (12, 13): pixel coordinate, 54: the 55th image
print(x_train[54, 12, 13, 1])

######################
# building the model #

method = 1

if method == 1:

    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(200, activation='relu'),
        Dense(150, activation='relu'),
        Dense(10, activation='softmax')
    ])

elif method == 2:

    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(10, activation='softmax'))

elif method == 3:

    input_layer = Input(shape=(32, 32, 3))
    x = input_layer
    x = Flatten()(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(150, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)
    model = Model(input_layer, output_layer)

show_structure(model, __file__)

#######################
# compiling the model #

opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

######################
# training the model #

model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)
#model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True, validation_data=(x_test, y_test))

########################
# evaluating the model #

model.evaluate(x_test, y_test, batch_size=1000)

######################
# drawing the result #

indices = select_indices(len(x_test), 10)
x = select_items(x_test, indices)
y = select_items(y_test, indices)

answer = to_classes(y, 'CIFAR10')

if method == 1:
    predicted = to_classes(model.predict(x), 'CIFAR10')
elif method == 2:
    predicted = to_classes(model.predict_proba(x), 'CIFAR10')
elif method == 3:
    predicted = CIFAR10_CLASSES[model.predict_classes(x)]

viewer = Viewer(num_rows=2, num_cols=10, width=15, height=2)
viewer.add_row(x, answer, predicted)
viewer.show()