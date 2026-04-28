# Dependencies,
import numpy as np

class PCA():

    def __init__(self, n_components=None):
        """Constructor method. We create our class variables here."""

        # Assigning parameters,
        self.n_components = n_components

        # Creating class attributes,
        self.means, self.cov_matrix, self.eigenvectors, self.eigenvalues, self.X = None, None, None, None, None
        self.X, self.N = None, None
        self.means = None

    def fit_transform(self, X):
        """Constructs the covariance matrix of X and solves the PCA eigenvector equation. These are used to determine the 
        principle components and apply the transformation. The principle components and the transformed data are returned."""

        # Assigning to class variables,
        self.X, self.N = X, X.shape[0]
        if self.n_components is None:
            self.n_components = X.shape[1]

        # Constructing covariance matrix,
        self.cov_matrix = self.compute_cov_matrix(X)

        # Centering, 
        X_centered = self.X - self.means

        # Solving eigenvalue equation,
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.cov_matrix)

        # Sorting eigenvalues,
        sorted_idxs = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues, self.eigenvectors = self.eigenvalues[sorted_idxs], self.eigenvectors[:, sorted_idxs]

        # Applying the transformation to our original dataset,
        X_projected = np.dot(X_centered, self.eigenvectors[:self.n_components].T)

        # Extracting our relevent principal components,
        prin_components = self.eigenvectors[:, :self.n_components]

        # Returning the relevent principle components,
        return prin_components, X_projected
    
    def transform(self, X):
        """Applys the transformation on a given dataset X."""

        # Centering the data, 
        X_centered = self.X - self.means

        # Applying the transformation,
        X_projected = np.dot(X_centered, self.eigenvectors[:self.n_components].T)

        return X_projected

    def compute_cov_matrix(self, X):

        # Subtracting mean,
        self.means = np.mean(X, axis=0)
        X_centered = X - self.means

        # Appplying formula,
        cov_matrix = np.matmul(X_centered.T, X_centered)/self.N

        return cov_matrix
    
    def compute_var_ratios(self):
        
        # Computing the variance ratio of each component,
        ratios = self.eigenvalues/np.sum(self.eigenvalues)

        return ratios[:self.n_components]
    
    def compute_compression_ratio(self):

        # Number of dimensions in original and reduced datasets,
        N, D, K = self.X.shape[0], self.X.shape[1], self.n_components

        # Computing compression ratio,
        c_ratio = 1 - (K*(N + D)/(N*D))

        return c_ratio