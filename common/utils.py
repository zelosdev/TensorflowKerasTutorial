############################
# turning off the warnings #

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

#####################
# importing modules #

import os
import numpy as np
import tensorflow as tf

def select_indices(num_items, num_to_select, random_seed=None):
    np.random.seed(random_seed)
    return np.random.choice(range(num_items), num_to_select)

# Both 'items' and 'indices' must be numpy arrays.
def select_items(items, indices):
    return items[indices]

CIFAR10_CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

CIFAR100_CLASSES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

def to_classes(y, dataset_name):
    if dataset_name == 'CIFAR10':
        return CIFAR10_CLASSES[np.argmax(y, axis=-1)]
    elif dataset_name == 'CIFAR100':
        return CIFAR100_CLASSES[np.argmax(y, axis=-1)]

def show_structure(model, file_name=None):
    model.summary()
    if file_name is not None:
        file_name = file_name.split('.')[0]
        file_name += '.png'

    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

def create_directory(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print('Failed to create directory:', path)

def get_bounding_box_2D(points):
    xMin, xMax = min(points[:, 0]), max(points[:, 0])
    yMin, yMax = min(points[:, 1]), max(points[:, 1])
    return xMin, xMax, yMin, yMax

def generate_random_points(num_points, xMin, xMax, yMin, yMax):
    x = np.random.uniform(xMin, xMax, size=num_points)
    y = np.random.uniform(yMin, yMax, size=num_points)
    return np.array(list(zip(x, y)))

def generate_uniform_grid_points(nx, ny, xMin, xMax, yMin, yMax):
    x = np.linspace(xMin, xMax, nx)
    y = np.linspace(yMin, yMax, ny)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    return np.array(list(zip(xv, yv)))