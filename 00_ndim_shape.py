##########################################
# to turn off warnings                   #
import warnings                          #
warnings.filterwarnings('ignore')        #
                                         #
import os                                #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #
                                         #
##########################################

import tensorflow as tf
from tensorflow.keras.layers import add, concatenate

print(tf.__version__) # 2.1.0
print(tf.keras.__version__) # 2.2.4-tf

x = tf.constant(3.14159)
print(x.ndim) # 0
print(x.shape) # () because it is not a tensor

x = tf.constant([3.14159])
print(x.ndim) # 1
print(x.shape) # (1,)

x = tf.constant([[3.14159]])
print(x.ndim) # 2
print(x.shape) # (1,1)

x = tf.constant([[[3.14159]]])
print(x.ndim) # 3
print(x.shape) # (1,1,1)

x = tf.constant([[1, 2], [3, 4], [5, 6]])
print(x.ndim) # 2
print(x.shape) # (3,2)
