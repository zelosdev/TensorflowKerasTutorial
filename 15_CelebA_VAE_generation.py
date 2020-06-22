from common.utils import *
from common.loaders import *
from common.VAE import *

# The original size of a CelebA image is 178 x 218.
# We would change the dimension to 128 x 128 using tensorflow.keras.preprocessing.image.ImageDataGenerator.

_, data_flow = load_celeba_images(path='./data/CelebA/', target_dim=(128, 128, 3), batch_size=32)

vae = VAE(
    input_dim                  = (128, 128, 3),    # the dimension of the input data
    z_dim                      = 200,              # the dimension of the latent space
    encoder_conv_filters       = [32, 64, 64, 64], # encoder parameters
    encoder_conv_kernel_size   = [ 3,  3,  3,  3],
    encoder_conv_strides       = [ 2,  2,  2,  2],
    decoder_conv_t_filters     = [64, 64, 32,  3], # decoder parameters
    decoder_conv_t_kernel_size = [ 3,  3,  3,  3],
    decoder_conv_t_strides     = [ 2,  2,  2,  2],
    use_batch_norm             = False,            # copiling parameters
    use_dropout                = False,
    export_path                = 'result/CelebA_VAE'
)

vae.compile(learning_rate=0.0005, rc_loss_factor=10000)
vae.train(data_flow, batch_size=32, epochs=200, with_generator=True) # It will take few hours!