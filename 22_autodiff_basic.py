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

#############
# example 1 #
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x) # If you want to differentiate about a constant tensor,
                  # you should use taple.watch() function
                  # to change the constant tensor like a variable tensor.
    y = (2 * x * x) + (3 * x) + 1

result = tape.gradient(y, x) # = dy/dx
print(result.numpy()) # 15.0

#############
# example 2 #
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x * x

result = tape.gradient(y, x)
print(result.numpy()) # 6.0

#############
# example 3 #
def diff_func(x):
    x = tf.Variable(x)
    with tf.GradientTape() as tape:
        if x <= 0:
            y = 2 * x * x
        else:
            y = 3 * x * x
    result = tape.gradient(y, x)
    return result.numpy()

print(diff_func(-3.0)) # -12.0
print(diff_func( 0.0)) #   0.0
print(diff_func( 3.0)) #  18.0

#############
# example 4 #
x = tf.Variable(2.0)
y = tf.Variable(4.0)

with tf.GradientTape() as tape:
    f = 2 * x + 3 * y

result = tape.gradient(f, [x, y])
print(result[0].numpy(), result[1].numpy()) # 2.0 3.0