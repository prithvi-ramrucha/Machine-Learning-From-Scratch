# Dependencies,
import numpy as np

class LinearRegression:
    """Class for linear regression."""

    def __init__(self, label=None):
        """Constructor method."""

        # Model attributes,
        self.params = []
        self.label = "linear regressor" if label is None else label
        self.fitted = False
        self.scored = False
        self.model_score = None

        # Training data,
        self.X, self.y = None, None
        self.n_samples, self.n_features = None, None
        self.model_score = None

    def fit(self, X, y):
        """Use this method to fit the model."""

        # Book-keeping,
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape[0], X.shape[1]

        # Computing model parameters,
        self.params = self._compute_params(X, y)

        # Update fitted state,
        self.fitted = True

        return None

    def predict(self, X):
        """This method returns the predictions when supplied with a data matrix."""
        return X @ self.params

    def score(self, X, y):
        """Computes the R^2 coefficient on the given samples provided."""

        # Computing model predictions,
        y_pred = self.predict(X)

        # Computing R^2 coefficient,
        y_mean = np.mean(y)
        SSR = np.sum((y - y_pred)**2)
        SST = np.sum((y - y_mean)**2)
        score = 1 - SSR/SST

        # Marking model as scored,
        self.model_score = score
        self.scored = True

        return score

    def _compute_params(self, X, y):
        """Computes the parameters of the multiple regression from solving the normal equations via the left Moore-penrose pseduo-inverse."""

        # Computing the left Moore-Penrose psuedo inverse of X,
        X_inv = np.linalg.pinv(X)

        # Computing the parameters via the solution to the normal equations,
        params = X_inv @ y

        # Returning the parameters,
        return params

    def _repr_html_(self):
        """Compact HTML GUI as the object representation in Jupyter Notebook."""
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
                LinearRegression
                <div style="margin-top:2px;">
                    <img src="linearregression_icon.png" alt="tree icon" width="30" height="30">
                </div>
            </div>

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
            html += f"<b>Score:</b> {round(self.model_score, 3) if self.scored == True else None}<br>"
            html += f"self.n_features:</b> {self.n_features}<br>"
            html += f"self.n_samples:</b> {self.n_samples}<br>"
        else:
            html += "<b>Status:</b> <span style='color:red;'>Not Fitted</span><br>"

        return html