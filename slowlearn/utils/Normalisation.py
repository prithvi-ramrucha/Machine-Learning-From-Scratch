# Dependencies,
import numpy as np

def min_max(X):
    """Performs min-max normalisation on a (n, M) shaped data X where n is the number of samples
    and M the number of features."""

    # Computing minimum and maximum values of each feature,
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)

    # Performing normalisation,
    X_normed = (X - X_min)/(X_max - X_min)

    return X_normed

def standardise(X):
    """Performs mean normalisation on a (n, M) shaped data X where n is the number of samples
    and M the number of features."""

    # Computing minimum and maximum values of each feature,
    X_std = np.std(X, axis=0)
    X_mean = np.mean(X, axis=0)

    # Performing normalisation,
    X_normed = (X - X_mean)/X_std

    return X_normed