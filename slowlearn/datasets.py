# Dependencies,
import numpy as np

class Dataset():
    """Class dataset objects. Acts a container for both the data and its metadata."""

    def __init__(self, X, y):
        """Constructor method for the class."""

        # Storing the data,
        self.X, self.y = X, y

        # Additional metadata,
        self.target_names = None
        self.feature_names = None
        self.n_samples = None
        self.feature_dims = None

    def add_metadata(self, target_names=None, feature_names=None, n_samples=None, feature_dims=None):
        """Adds the metadata to the dataset object."""

        self.target_names = target_names
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.feature_dims = feature_dims

def load_MNIST():
    """This method loads the MNIST dataset and returns it as a dataset object."""

    # Loading data as NumPy arrays,
    X, y = np.load("data/MNIST-X.npy"), np.load("data/MNIST-y.npy")

    # Creating the dataset object,
    dataset = Dataset(X, y)
    dataset.add_metadata(
        target_names=["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"],
        feature_names=None,
        n_samples=60000,
        feature_dims=(28, 28)
    )

    # Returning the dataset object,
    return dataset