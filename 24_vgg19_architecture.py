##########################################
# to turn off warnings                   #
import warnings                          #
warnings.filterwarnings('ignore')        #
                                         #
import os                                #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #
                                         #
##########################################

from tensorflow.keras.applications import vgg19

model = vgg19.VGG19(weights="imagenet", include_top=False)
model.summary()

layer_outputs = dict([(layer.name, layer.output) for layer in model.layers])

for key in layer_outputs.keys():
    print(key, ': ', layer_outputs[key])