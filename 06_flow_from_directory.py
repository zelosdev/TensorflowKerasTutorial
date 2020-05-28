from common.utils import *
from common.Viewer import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(
    rotation_range     = 20,         # rotation
    width_shift_range  = 0.2,        # horizontal translation
    height_shift_range = 0.2,        # vertical translation
    rescale            = 1.0 / 255.0 # rescale: from [0, 255] to [0.0, 1.0]
)

batch_size = 6 # to read 6 images at once
iterations = 5 # to augment the data 5 times

data_flow = data_gen.flow_from_directory(
    directory   = './data/animals', # data directory path
    target_size = (100, 100), # from (128, 128) to (100, 100) (NOTE: no channels)
    batch_size  = batch_size, # batch size specifying how many images to be read at once
    shuffle     = True, # random shuffling
    class_mode  = 'categorical' # to get the one-hot encoded labels
    #class_mode  = 'binary' # to get the single digit labels
)

######################
# iteration method 1 #
viewer = Viewer(num_rows=iterations, num_cols=batch_size, width=10, height=10)

for i in range(iterations):

    images, labels = [], []

    image, label = data_flow.next()

    for j in range(batch_size):
        images.append(image[j])
        labels.append(label[j])

    viewer.add_row(images, labels)

viewer.show()

######################
# iteration method 2 #
viewer = Viewer(num_rows=iterations, num_cols=batch_size, width=10, height=10)

for i, (image, label) in enumerate(data_flow):

    images, labels = [], []

    for j in range(batch_size):
        images.append(image[j])
        labels.append(label[j])

    viewer.add_row(images, labels)

    if i == iterations-1:
        break

viewer.show()