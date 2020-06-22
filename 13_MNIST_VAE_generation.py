from common.utils import *
from common.loaders import *
from common.VAE import *

(x_train, y_train), (x_test, y_test) = load_mnist()

vae = VAE(
    input_dim                  = (28, 28, 1),      # the dimension of the input data
    z_dim                      = 2,                # the dimension of the latent space
    encoder_conv_filters       = [32, 64, 64, 64], # encoding parameters
    encoder_conv_kernel_size   = [ 3,  3,  3,  3],
    encoder_conv_strides       = [ 1,  2,  2,  1],
    decoder_conv_t_filters     = [64, 64, 32,  1], # decoding parameters
    decoder_conv_t_kernel_size = [ 3,  3,  3,  3],
    decoder_conv_t_strides     = [ 1,  2,  2,  1],
    use_batch_norm             = True,
    use_dropout                = True,
    export_path                = 'result/MNIST_VAE'
)

vae.compile(learning_rate=0.0005, rc_loss_factor=1000)
vae.train(x_train, batch_size=1000, epochs=200)