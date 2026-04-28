# Dependencies,
import numpy as np

class KNNClassifier():
    """The class for the k-nearest neighbour algorithm for classifications."""

    def __init__(self, k, label=None):
        """Constuctor method."""

        # Hyperparameter
        self.k = k

        # Class attributes,
        self.label = label if label is not None else "untitled"
        self.X, self.y = None, None
        self.n_features, self.n_samples = None, None
        self.model_score = None
        self.fitted = False
        self.scored = False

    def fit(self, X, y):
        """KNNs do not require fitting in the traditional sense. We keep this method for consistency with other models."""

        # Assigning class attributes,
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape[0], X.shape[1]
        self.fitted = True

        # Sanity check,
        if self.k > self.n_samples:
            raise ValueError("The number of closest neighbours k cannot be larger than number of training samples self.n_samples.")

    def predict(self, X):
        """"""

        # Constructing distance matrix,
        D = self._L2_squared(X)

        # Finding k-nearest neighbours for each unseen sample,
        idxs = np.argpartition(D, (self.k - 1), axis=-1)[:, :self.k]
        k_labels = self.y[idxs]

        # Taking the majority vote of the nearest neighbours,
        preds = np.argmax(np.apply_along_axis(np.bincount, axis=1, arr=k_labels, minlength = np.max(k_labels)+1), axis=1)

        return preds

    def score(self, X, y):
        """Returns the classification accuracy of the model on a set of given samples."""

        # Computing predictions,
        y_pred = self.predict(X)

        # Calculating the classification accuracy,
        model_score = np.mean(y_pred == y)
        self.model_score = model_score
        self.scored = True

        return model_score

    def _L2_squared(self, X):
        """Computes the distance matrix which contains the pair-wise squared euclidean distances between all unseen and seen training data points."""

        # Converting samples into NumPy array,
        X, Q = np.array(X), self.X

        # Computing norm vectors,
        q_norm = np.sum(X**2, axis=1)
        x_norm = np.sum(Q**2, axis=1)

        # Constructing distance matrix,
        D = q_norm[:, None] + x_norm[None, :] -2*(X @ Q.T)

        return D

    def _repr_html_(self):
        """A compact HTML GUI as the object representation in Jupyter Notebook."""
        html = f"""
        <div style="
            border:1px solid black;
            border-radius:6px;
            font-family:Arial, sans-serif;
            font-size:12px;
            line-height:1.2;
            width:fit-content;
            background:white;
            color:black;
            padding-left:8px;
            padding-right:8px;
        ">
            <!-- Title bar -->
            <i>{self.label}</i>
            <div style="
                background:#e0e0e0;
                padding:3px 6px;
                font-weight:bold;
                border-bottom:1px solid black;
                border-top-left-radius:6px;
                border-top-right-radius:6px;
                color:black;
            ">
                KNNClassifier
            </div>

            <!-- Hyperparameters -->
            <ul style="margin:4px 0 4px 16px; padding:0;">
                <b>Hyperparameters:</b><br>
                k:</b> {self.k}<br>
            </ul>

            <!-- Divider -->
            <div style="
                border-top:1px solid #ccc;
                margin:4px 0;
            "></div>

            <!-- Status and other info -->
            <ul style="margin:4px 0 4px 16px; padding:0;">
        """

        if self.fitted:
            html += "<b>Status:</b> <span style='color:green;'>Fitted</span><br>"
            html += f"Score:</b> {round(self.model_score, 3) if self.scored else None}<br>"
            html += f"n_samples:</b> {self.n_samples}<br>"
            html += f"n_features:</b> {self.n_features}<br>"
        else:
            html += "<b>Status:</b> <span style='color:red;'>Not fitted</span>"

        html += "</ul></div>"
        return html