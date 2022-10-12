import numpy as np


def add_gaussian_noise(array, percent_of_noised_array=0.001):
    """
    Args:
        array : numpy array
    Return :
        array : numpy array with gaussian noise added
    """
    mean = np.nanmean(array)
    std = np.nanstd(array)

    gaus_noise = np.random.normal(mean, std, array.shape)
    gaus_noise = np.where(gaus_noise < 0, gaus_noise, abs(gaus_noise) )

    indices = np.random.choice(gaus_noise.shape[1] * gaus_noise.shape[0], replace=False,
                               size=int(gaus_noise.size * (1 - percent_of_noised_array)))
    gaus_noise[np.unravel_index(indices, gaus_noise.shape)] = 0
    noised_array = array + gaus_noise
    # print(np.count_nonzero(gaus_noise)/gaus_noise.size)

    return noised_array