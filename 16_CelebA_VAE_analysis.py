from common.utils import *
from common.loaders import *
from common.Viewer import *
from common.VAE import *

import pandas as pd
from scipy.stats import norm

vae = load_model(VAE, 'result/CelebA_VAE')

# reading the attributes from the .csv file
attr = pd.read_csv('C:/work/KerasTest/data/CelebA/list_attr_celeba.csv')
#print(attr[:10]['5_o_Clock_Shadow'])
#print(attr['image_id'])

data_gen = ImageDataGenerator(rescale=1./255.)

data_flow = data_gen.flow_from_dataframe(
    dataframe   = attr, # attributes
    directory   = 'C:/work/KerasTest/data/CelebA/img_align_celeba',
    x_col       = 'image_id',
    y_col       = 'Eyeglasses', # Take this column only and return it to the label.
    target_size = (128, 128),
    class_mode  = 'other', # numpy array of y_col data
    batch_size  = 1000,
    shuffle     = True
)

# reconstructed faces, after passing through the encoder and decoder
def show01():

    input_images, _ = data_flow.next() # Get the first batch.
    input_images = input_images[:10] # Select the first 10 images from it.

    # method 1
    #z_points = vae.encoder.predict(input_images)
    #reconstructed_images = vae.decoder.predict(z_points)
    #print(len(z_points)) = 10: # of inputs
    #print(z_points.shape) = (10, 200): (# of inputs, z_dim)

    # method 2
    output_images = vae.model.predict(input_images)

    viewer = Viewer(num_rows=2, num_cols=10, width=10, height=2)
    viewer.add_row(input_images)
    viewer.add_row(output_images)
    viewer.show()

# distributions of points for the first 50 dimensions in the latent space
def show02():

    z_test = vae.encoder.predict_generator(data_flow, steps=20, verbose=1)
    #print(z_test.shape) # (200, 200)

    x = np.linspace(-3, 3, 100)

    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    for i in range(50):
        ax = fig.add_subplot(5, 10, i+1)
        ax.hist(z_test[:,i], density=True, bins=30)
        ax.axis('off')
        ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
        ax.plot(x, norm.pdf(x))

    plt.show()

# newly generated faces
def show03():

    viewer = Viewer(num_rows=3, num_cols=10, width=10, height=3)

    for i in range(3):
        z_points = np.random.normal(size=(10, vae.z_dim))
        generated_images = vae.decoder.predict(np.array(z_points))
        viewer.add_row(generated_images)

    viewer.show()

# faces that satisfy the conditions
def show04():

    viewer = Viewer(num_rows=5, num_cols=10, width=10, height=5)

    for i in range(5):

        satisfied_images = []

        while True:

            images, labels = data_flow.next()

            for j in range(len(images)):
                if labels[j] == 1: # satisfied!
                    satisfied_images.append(images[j])

            if len(satisfied_images) > 10:
                break

        viewer.add_row(satisfied_images[:10]) # Show only the first 10 images in a column.

    viewer.show()

# adding and subtracting features to and from faces
def show05():

    z_avg_pos = np.zeros(shape=vae.z_dim, dtype='float32') # for the attribute with '+1'
    z_avg_neg = np.zeros(shape=vae.z_dim, dtype='float32') # for the attribute with '-1'

    z_count_pos = 0
    z_count_neg = 0

    while True:

        images, labels = data_flow.next()

        z_points = vae.encoder.predict(images)

        z_points_pos = z_points[labels== 1]
        z_points_neg = z_points[labels==-1]

        z_avg_pos += np.sum(z_points_pos, axis=0)
        z_avg_neg += np.sum(z_points_neg, axis=0)

        z_count_pos += len(z_points_pos)
        z_count_neg += len(z_points_neg)

        if z_count_pos > 2000: # 2000 points are enough to get the center.
            break;

    z_avg_pos = z_avg_pos / z_count_pos
    z_avg_neg = z_avg_neg / z_count_neg

    z_direction = z_avg_pos - z_avg_neg
    magnitude = np.linalg.norm(z_direction) # magnitude
    z_direction = z_direction / magnitude # normalization

    # Select the first five images from the next batch.
    images, _ = data_flow.next()[:5]

    z_points = vae.encoder.predict(images)

    z_scales = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    viewer = Viewer(num_rows=5, num_cols=9, width=9, height=5)

    for i in range(5):

        images = []

        for z_scale in z_scales:

            changed_z_point = z_points[i] + z_direction * z_scale
            changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]
            images.append(changed_image)

        viewer.add_row(images)

    viewer.show()

# morphing between two faces
def show06():

    start_image_file = '000238.jpg'
    end_image_file = '000193.jpg'

    images, _ = data_flow.next()

    z_points = vae.encoder.predict(images)

    interpolated_images = []

    interpolated_images.append(images[0]) # Add the first selected image.

    factors = np.arange(0, 1, 0.1)

    for factor in factors:

        z = z_points[0] * (1 - factor) + z_points[1]  * factor
        img = vae.decoder.predict(np.array([z]))[0]
        interpolated_images.append(img)

    interpolated_images.append(images[1]) # Add the second selected image.

    num_interpolated_images = len(factors)

    viewer = Viewer(num_rows=1, num_cols=num_interpolated_images+2, width=num_interpolated_images+2, height=1)
    viewer.add_row(interpolated_images)
    viewer.show()

show01()
show02()
show03()
show04()
show05()
show06()