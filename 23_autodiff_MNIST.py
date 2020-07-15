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
import tensorflow as tf

##############
# MNIST data #

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = tf.expand_dims(x_train, -1) # from (60000, 28, 28) to (60000, 28, 28, 1)
x_test = x_test.astype('float32') / 255.0
x_test = tf.expand_dims(x_test, -1) # from (10000, 28, 28) to (10000, 28, 28, 1)

###########
# dataset #

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#########
# model #

input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(input_layer)
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(input_layer, output_layer)
#model.summary()

#########################################
# loss function, optimizer, and metrics #

loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

test_loss = tf.keras.metrics.Mean()
test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

############################
# train/test step function #

# @tf.function
# This decoration converts a Python function to its graph representation.
# In general, it's not necessary to decorate each of these smaller functions with tf.function.
# Only use tf.function to decorate high-level computations
# for example, one step of training or the forward pass of your model.

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape: # to differentiate
        predictions = model(images)
        loss = loss_function(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss) # loss update
    train_acc(labels, predictions) # accuracy update

@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_function(labels, predictions)

    # no need to compute the gradient and backpropagation for the test set

    test_loss(loss) # loss update
    test_acc(labels, predictions) # accuracy update

EPOCHS = 5

for epoch in range(EPOCHS):

    for images, labels in train_dataset:
        train_step(images, labels)

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    template = 'Epoch: {}, Train-Loss: {:.5f}, Train-Accuracy: {:.2f}%, Test-Loss: {:.5f}, Test-Accuracy: {:.2f}%'
    print (template.format(epoch+1, train_loss.result(), train_acc.result()*100, test_loss.result(), test_acc.result()*100))