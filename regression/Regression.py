import numpy as np


class Regression:

    def __init__(self, alpha=0, ratio=0):
        self.alpha = alpha
        self.ratio = ratio
        self.lr = None
        self.epochs = None
        self.X = None
        self.y = None
        self.w = None
        self.m = None
        self.n = None

    def reset_parameters(self, lr=0.00002, epochs=10000, alpha=0, ratio=0):
        self.alpha = alpha
        self.ratio = ratio
        self.lr = lr
        self.epochs = epochs

    def compute_cost(self, y_hat):
        regularization = self.alpha * (
                self.ratio * np.sum(np.abs(self.w)) + 0.5 * (1 - self.ratio) * np.sum(np.square(self.w)))
        return (1 / (2 * self.m)) * np.sum(np.square(y_hat - self.y)) + regularization

    def update_parameters(self, y_hat):
        l1_derivation = self.alpha * self.ratio * np.sign(self.w)
        l2_derivation = self.alpha * (1 - self.ratio) * self.w
        dw = (1 / self.m) * np.dot(self.X.T, ( self.y-y_hat)) + l1_derivation + l2_derivation
        self.w = self.w - self.lr * dw

    def fit(self, X, y, lr=00000.2, epochs=10000):
        self.X = np.insert(X, 0, 1, axis=1)
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.m, self.n = self.X.shape
        self.w = np.random.rand(self.n,1)
        losses = []
        for i in range(epochs):
            y_hat = np.dot(self.X, self.w).reshape((self.m,1))
            cost = self.compute_cost(y_hat)
            losses.append(cost)
            self.update_parameters(y_hat)

        return losses

    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        y_pred = np.dot(X_test, self.w)
        return y_pred
