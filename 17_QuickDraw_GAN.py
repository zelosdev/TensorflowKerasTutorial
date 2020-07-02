from common.utils import *
from common.loaders import *
from common.GAN import *

x_train = load_quickdraw_camel()

print(x_train.shape)
plt.imshow(x_train[200,:,:,0], cmap = 'gray')
plt.show()

gan = GAN(
    input_dim                  = (28, 28, 1),
    z_dim                      = 100,
    d_conv_filters             = [64, 64, 128, 128],
    d_conv_kernel_size         = [ 5,  5,   5,   5],
    d_conv_strides             = [ 2,  2,   2,   1],
    d_batch_norm_momentum      = None,
    d_dropout_rate             = 0.4,
    g_initial_dense_layer_size = (7, 7, 64),
    g_upsample                 = [2, 2, 1, 1],
    g_conv_filters             = [128, 64, 64, 1],
    g_conv_kernel_size         = [  5,  5,  5, 5],
    g_conv_strides             = [  1,  1,  1, 1],
    g_batch_norm_momentum      = 0.9,
    g_dropout_rate             = None
)

gan.compile(d_learning_rate=0.0008, g_learning_rate=0.0004)
gan.train(x_train, batch_size=256, epochs=6000)