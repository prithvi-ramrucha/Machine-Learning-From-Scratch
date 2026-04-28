# Dependencies,
import numpy as np

def strat_sample(X, y, ratios, seed):
    """Performs stratified sampling on samples given by a data matrix X."""

    # Setting global seed,
    np.random.seed(seed=seed)

    # Permutating the sample indices,
    n_samples = len(X)
    perm = np.random.permutation(n_samples)

    # Computing strata sizes,
    train_ratio, test_ratio = ratios
    test_size = int(test_ratio*n_samples)
    train_size = int(train_ratio*n_samples)

    # Slicing,
    test_idxs = perm[:test_size]
    train_idxs = perm[test_size: (test_size + train_size)]
    val_idxs = perm[(test_size + train_size):] # <-- The remaining samples.

    X_train, y_train = X[train_idxs], y[train_idxs]
    X_test, y_test = X[test_idxs], y[test_idxs]
    X_val, y_val = X[val_idxs], y[val_idxs]

    return X_train, X_test, X_val, y_train, y_test, y_val