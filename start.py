import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import load_sample_image




def compare_images(img, img_compressed, k):
    """Show the compressed and uncompressed image side by side.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    axes[0].set_axis_off()
    if isinstance(k, str):
        axes[0].set_title(k)
    else:
        axes[0].set_title(f"Compressed to {k} colors")
    axes[0].imshow(img_compressed)
    axes[1].set_axis_off()
    axes[1].set_title("Original")
    axes[1].imshow(img)
    plt.show()


X = np.array(Image.open("/Users/xisun/Documents/MATLAB/images/IMG_2049.jpeg"))


def kmeans(X, k):


    N, D = X.shape

    # Pick k random points to initialize mu
    mu = X[np.random.choice(N, size=k, replace=False)].astype(np.float)

    z = np.empty(N, dtype=np.int)
    J_prev = None
    while True:
        # Update the cluster indicators
        z = ((X[:, np.newaxis] - mu[np.newaxis]) ** 2).sum(axis=-1).argmin(axis=-1)

        # Count data points per cluster
        N_k = np.bincount(z, minlength=k)

        # Restart if we lost any clusters
        if np.any(N_k == 0):
            mu = X[np.random.choice(N, size=k, replace=False)]
            J_prev = None
            continue

        # Update centroids
        for i in range(k):
            mu[i] = X[z == i].sum(axis=0)
        mu = mu / N_k[:, np.newaxis]

        # Recompute the distortion
        J = ((X - mu[z]) ** 2).sum()

        # Check for convergence
        # 也就是说X里面每个点到自己对应的中心的距离的和必须每一个loop都要降低，一旦距离的和不降低了，就结束loop
        if J_prev is None or J < J_prev:
            J_prev = J
        else:
            break

    return mu, z


# Cluster the color values
k = 5
mu, z = kmeans(X.reshape((-1, 3)), k)

# Replace each pixel with its cluster color
X_compressed = mu[z].reshape(X.shape).astype(np.uint8)

# Show the images side by side
compare_images(X, X_compressed, k)