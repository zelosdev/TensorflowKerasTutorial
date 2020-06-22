#####################
# importing modules #

from common.utils import *
from common.loaders import *
from common.Viewer import *
from common.AE import *

########################
# loading data & model #

(_, _), (x_test, y_test) = load_mnist()
ae = load_model(AE, 'result/MNIST_AE')

# all points
z_points = ae.encoder.predict(x_test)
xMin, xMax, yMin, yMax = get_bounding_box_2D(z_points)

##########################################
# drawing the result - 30 random samples #

plt.figure(figsize=(12, 12))
plt.scatter(z_points[:, 0] , z_points[:, 1], cmap='rainbow', c=y_test, alpha=0.5, s=2)
plt.colorbar()

num_samples = 30
z_samples = generate_random_points(num_samples, xMin, xMax, yMin, yMax)
plt.scatter(z_samples[:, 0] , z_samples[:, 1], c='black', alpha=1, s=20)

# reconstructed images
reconstructed_images = ae.decoder.predict(z_samples)
z_samples = np.round(z_samples, 1)
viewer = Viewer(num_rows=3, num_cols=10, width=15, height=3)
viewer.add_row(reconstructed_images[0:10], z_samples[0:10])
viewer.add_row(reconstructed_images[10:20], z_samples[10:20])
viewer.add_row(reconstructed_images[20:30], z_samples[20:30])
viewer.show()

plt.show()

##############################################
# drawing the result - 20x20 uniform samples #

plt.figure(figsize=(12, 12))
plt.scatter(z_points[:, 0] , z_points[:, 1], cmap='rainbow', c=y_test, alpha=0.5, s=2)
plt.colorbar()

nx = ny = 20
z_samples = generate_uniform_grid_points(nx, ny, xMin, xMax, yMin, yMax)
plt.scatter(z_samples[:, 0] , z_samples[:, 1], c='black', alpha=1, s=20)

reconstructed_images = ae.decoder.predict(z_samples)
viewer = Viewer(num_rows=ny, num_cols=nx, width=12, height=12)
for i in range(ny):
    viewer.add_row(reconstructed_images[i*nx: (i+1)*nx])
viewer.show()