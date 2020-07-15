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
from tensorflow import keras
from tensorflow.keras.applications import vgg19

base_image_path = 'data/style_transfer/tuebingen.jpg' # 1024 x 768
style_reference_image_path = 'data/style_transfer/starry_night.jpg' # 512 x 344

#base_image_path = 'data/style_transfer/bullfight.jpg' # 700 x 277
#style_reference_image_path = 'data/style_transfer/leejungseob_white_ox.jpg' # 810 x 6154

total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

result_prefix = 'result/style_transfer/result' # 533 x 400

width, height = keras.preprocessing.image.load_img(base_image_path).size

# image resolution to be handled the network and for the output
img_nrows = 400 # the height of the output image
img_ncols = int(width * img_nrows / height) # width : height = img_cols : img_rows

def load_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = keras.preprocessing.image.img_to_array(img) # from 'PIL.JpegImagePlugin.JpegImageFile' to 'numpy.ndarray'
    img = np.expand_dims(img, axis=0) # from (400, 533, 3) to (1, 400, 533, 3)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(base, output):
    return tf.reduce_sum(tf.square(output - base))

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1)) 
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, output):
    S = gram_matrix(style)
    C = gram_matrix(output)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def total_variation_loss(x): # x: the generated image (=output image)
    # the squared difference between the original image the image translated horizontally
    a = tf.square(x[:, :img_nrows-1, : img_ncols-1, :] - x[:, 1:, :img_ncols-1, :])
    # the squared difference between the original image the image translated vertically
    b = tf.square(x[:, :img_nrows-1, : img_ncols-1, :] - x[:, :img_nrows-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

model = vgg19.VGG19(weights='imagenet', include_top=False) # the instance of the VGG19 model

layer_outputs = dict([(layer.name, layer.output) for layer in model.layers])
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer_outputs)
content_layer_name = "block5_conv2" # We choose the second convolutional layer of the fifth block to calculate the content loss.
style_layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"] # the list of layers to be used for the style loss.

def compute_loss(base_image, style_reference_image, generated_image):

    input_tensor = tf.concat([base_image, style_reference_image, generated_image], axis=0)
    features = feature_extractor(input_tensor)

    loss = tf.zeros(shape=()) # loss initialization

    # content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    output_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features, output_features)

    # style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        output_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, output_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # total variation loss
    loss += total_variation_weight * total_variation_loss(generated_image)
    return loss

base_image = load_image(base_image_path)
style_reference_image = load_image(style_reference_image_path)
generated_image = tf.Variable(load_image(base_image_path))

optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0,  decay_steps=100, decay_rate=0.96
    )
)

ITERATIONS = 4000
for i in range(1, ITERATIONS+1):

    with tf.GradientTape() as tape:
        loss = compute_loss(base_image, style_reference_image, generated_image)

    gradients = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])

    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        img = deprocess_image(generated_image.numpy())
        fname = result_prefix + "_at_iteration_%d.png" % i
        keras.preprocessing.image.save_img(fname, img)