############################
# turning off the warnings #

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import add, concatenate

x = tf.constant([[1, 2], [3, 4]])
print(x)
#tf.Tensor(
#[[1 2]
# [3 4]], shape=(2, 2), dtype=int32)

y = tf.constant([[10, 20], [30, 40]])
print(y)
#tf.Tensor(
#[[10 20]
# [30 40]], shape=(2, 2), dtype=int32)

print(concatenate([x, y], axis=0))
#tf.Tensor(
#[[ 1  2]
# [ 3  4]
# [10 20]
# [30 40]], shape=(4, 2), dtype=int32)

print(concatenate([x, y], axis=1))
#tf.Tensor(
#[[ 1  2 10 20]
# [ 3  4 30 40]], shape=(2, 4), dtype=int32)

print(add([x, y]))
#tf.Tensor(
#[[11 22]
# [33 44]], shape=(2, 2), dtype=int32)