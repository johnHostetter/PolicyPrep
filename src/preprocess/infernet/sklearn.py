from sklearn.base import BaseEstimator, TransformerMixin


class Flatten(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(-1, X.shape[-1])


class Reshape(BaseEstimator, TransformerMixin):
    def __init__(self, shape):
        self.shape = shape

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(self.shape)
