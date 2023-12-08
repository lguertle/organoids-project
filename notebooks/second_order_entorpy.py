import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import sklearn.neighbors


def calculate_gradient_img(img):
    """
    Calculate the magnitude of the gradient based on the
    gaussian derivatives in x and y.
    """
    sigma = 20
    dx = scipy.ndimage.gaussian_filter(img, sigma, order=[0, 1], mode="nearest")
    dy = scipy.ndimage.gaussian_filter(img, sigma, order=[1, 0], mode="nearest")
    img = np.sqrt(dx**2 + dy**2)
    return img


# The maximum value to be used of the probabilites fed to
# to the entropy function
max_value = 256

initial_path = "E:/mouse_organoids_laure"


fig, axes = plt.subplots(4, 2, sharex=True, figsize=(10, 10))
# Iterate over the classes of images that are supposed to have different entropy
for col, folder in enumerate(["empty_png", "j8_png"]):
    # Iterate over all of the images for each class
    full_folder_path = pathlib.Path(initial_path) / folder
    for row, path in enumerate(full_folder_path.glob("*.png")):
        img = skimage.io.imread(path)
        img = img.astype(float)
        img = calculate_gradient_img(img)
        skimage.io.imsave(f"{folder}_{path.stem}_derivative.tif", img)
        values = img.flatten()
        # The kernel density estimate allows us to approximate a probability distribtuion
        # for the pixel values and does not contain zeros
        kde = sklearn.neighbors.KernelDensity(kernel="exponential", bandwidth=1)
        # Choose a random subset of pixels to speed up the fitting of the kernel density estimate
        kde_values = np.random.choice(values, size=10000, replace=False)
        # Set the range of possible pixel values
        x_range = (0, max_value)
        kde.fit(kde_values[:, None])
        # Plot a histrogram of pixel values and the estimated probability density function
        axes[row, col].hist(values, range=x_range, bins=100, density=True, label="Hist")
        x = np.arange(*x_range)
        pks = np.exp(kde.score_samples(x[:, None]))
        axes[row, col].plot(x, pks, label="KDE")
        # Compute the entropy based on the probabilities for all possible pixel values
        entropy = scipy.stats.entropy(pks)
        axes[row, col].set_title(path.stem)
        h = max(pks) * 0.8
        axes[row, col].text(120, h, f"H={entropy:.2f}")
        axes[row, col].set_ylabel("Prob.")
        print(folder, path.stem, entropy, img.min(), img.max())
        axes[3, 0].set_xlabel("Derivative value")
        axes[3, 1].set_xlabel("Derivative value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder}_{path.stem}_second_order_entropy.pdf")
