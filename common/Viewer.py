import matplotlib.pyplot as plt

class Viewer():

    def __init__(self, num_rows, num_cols, width=15, height=3):

        self.num_rows = num_rows
        self.num_cols = num_cols

        self.fig = plt.figure(figsize=(width, height))
        self.fig.subplots_adjust(hspace=0.4, wspace=0.4)

        self.index = 0

    def add_row(self, images, *labels):

        for j in range(self.num_cols):
            img = images[j]
            ax = self.fig.add_subplot(self.num_rows, self.num_cols, self.index+1)
            ax.axis('off')

            for i in range(len(labels)):
                ax.text(0.5, -0.35*(i+1), str(labels[i][j]), fontsize=10, ha='center', transform=ax.transAxes)

            if img.shape[2] == 3: # RGB image (ex. CIFAR-10)
                plt.imshow(img)
            if img.shape[2] == 1: # grayscale image (ex. MNIST)
                w, h = img.shape[0], img.shape[1]
                plt.imshow(img.reshape((w, h)), cmap='gray')

            self.index += 1

    def show(self):
        plt.show()

    def save(self, file_name):
        self.fig.savefig(file_name)
        plt.close()