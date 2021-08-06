import numpy as np
from skimage import io
from skimage.filters import laplace, gaussian
# from skimage.filters.edges import convolve
from scipy.ndimage import convolve
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
from utility import plot_mask


def mask_to_weighted(mask, w=10, b=1, sigma=3, method="gaussian"):
    res = mask.copy()
    res = laplace(res, ksize=3)
    res = (res != 0)
    res = res.astype(np.float32)
    if method == "gaussian":
        return w * gaussian(res, sigma) + b
    elif method == "distance":
        res = distance_transform_edt(1 - res)
        return w * np.exp(-res ** 2 / (2 * sigma ** 2)) + b
    else:
        raise ValueError("Please select method as either 'gaussian' or distance'.")


if __name__ == '__main__':
    mask_2 = r'image/mask_single_2.tif'
    mask = io.imread(mask_2)

    # res = mask.copy().astype(np.float32)
    # res1 = convolve(res, weights=[[0, .25, 0],
    #                               [.25, -1, .25],
    #                               [0, .25, 0]])
    # res2 = res1 != 0
    # res3 = gaussian(res2, sigma=3) * 20 + 1
    # plt.imshow(res3)
    weight = mask_to_weighted(mask, w=10, b=1, method='gaussian')
    # plt.subplot(121)
    # plt.imshow(mask)
    # plot_mask(mask)
    #
    # # plt.subplot(132)
    # # plt.imshow(res, cmap=plt.cm.inferno)
    #
    # plt.subplot(122)
    plt.imshow(weight, cmap=plt.cm.inferno, alpha=1)
    #
    plt.show()

