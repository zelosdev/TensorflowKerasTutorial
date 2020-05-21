from common.utils import *
from common.Viewer import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas
labels = pandas.read_csv('./data/Cifar-10/trainLabels.csv', dtype=str)

def append_png(file_name):
    return (file_name + '.png')

labels["id"] = labels["id"].apply(append_png) # ex) '1' -> '1.png'

for i in range(5):
    print(labels.id[i], labels.label[i])

data_gen = ImageDataGenerator(rescale=1./255., validation_split=0.25)

data_flow = data_gen.flow_from_dataframe(
    dataframe   = labels,                  # labels
    directory   = './data/Cifar-10/train', # path to the directory which contains all the images.
    x_col       = 'id',                    # the name of the column which contains the filenames of the images
    y_col       = 'label',                 # If class_mode is not 'raw' or not 'input' you should pass the name of the column which contains the class names.
                                           # None, if used for test_generator.
    subset      = 'training',              # This is for the training.
    batch_size  = 6,                       # batch size: the number of input data that will be propagated through the network at once
    shuffle     = True,                    # to shuffle or not
    class_mode  = 'categorical',           # 'categorical'(default), 'binary', 'sparse', 'input', 'raw', 'other', None
    target_size = (32, 32)                 # image resolution
)

viewer = Viewer(num_rows=5, num_cols=6, width=10, height=10)

for i in range(5):

    # method 1
    images, labels = data_flow.next()

    # method 2
    images, labels = next(data_flow)

    # method 3
    batch = next(data_flow)
    images = batch[0]
    labels = batch[1]

    classes = to_classes(labels, 'CIFAR10')
    viewer.add_row(images, classes)

viewer.show()