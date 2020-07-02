#####################
# importing modules #

from common.utils import *
from common.loaders import *
from common.Viewer import *

from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
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

######################
# training the model #

epochs = 10
batch_size = 1024

def next_batch(images, labels, batch_size):
    indices = select_indices(len(images), batch_size)
    return select_items(images, indices), select_items(labels, indices)

batches = int(len(x_train) / batch_size)

for epoch in range(epochs):
    for batch in range(batches):
        x_batch, y_batch = next_batch(x_train, y_train, batch_size)
        ret = model.train_on_batch(x_batch, y_batch) # the return value: (loss, accuracy) pair
        loss, accuracy = ret[0], ret[1]
        print('Epoch', epoch, ': Batch', batch, '-', model.metrics_names[0], '=', loss, '-', model.metrics_names[1], '=', accuracy)

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