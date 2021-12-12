from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

import numpy as np


class Perceptron(object):

    def __init__(self, eta=0.01, iterations=10):
        self.eta = eta
        self.iterations = iterations

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.iterations):
            errors = 0
            for xi, target in zip(X, y):
                output = self.predict(xi)
                update = self.eta * (target - output)
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

    def net_input(self, x):
        return np.dot(x.T, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0, 1, -1)


class Adaline(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum()/2.0
            self.cost_.append(cost)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

class MLP(object):

    def __init__(self,inputSize=7,  hiddenSize=5, outputSize=2, n_iter=1000, eta=0.01, u=0.3):
        # Define Hyperparameters
        self.inputLayerSize = inputSize
        self.hiddenLayerSize = hiddenSize
        self.outputLayerSize = outputSize
        self.n_iter = n_iter
        self.eta = eta
        self.u = u
        self.cost = []

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))
        # return (np.exp(z)-np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)
        # return (1 - self.sigmoid(z) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    def shuffle(self, X, y):
        r = np.random.permutation(y.shape[0])
        return X[r], y[r]

    def costFunctionPrime(self, X, y):

        w1_old = 0
        w2_old = 0
        self.cost = []


        for i in range(self.n_iter):
            # Compute derivative with respect to W and W2 for a given X and y:

            X, y = self.shuffle(X, y)

            self.yHat = self.forward(X)

            delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
            dJdW2 = np.dot(self.a2.T, delta3)

            delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
            dJdW1 = np.dot(X.T, delta2)

            self.W1 = self.W1 - self.eta * dJdW1 - self.u * (self.W1 - w1_old)
            self.W2 = self.W2 - self.eta * dJdW2 - self.u * (self.W2 - w2_old)

            w1_old = self.W1.copy()
            w2_old = self.W2.copy()

            self.cost.append(self.costFunction(X, y).sum())

    def rbf_kernel_pca(self, X, gamma, n_components):

        # Calculate pairwise squared Euclidean distances
        # in the MxN dimensional dataset.
        sq_dists = pdist(X, 'sqeuclidean')
        # Convert pairwise distances into a square matrix.
        mat_sq_dists = squareform(sq_dists)
        # Compute the symmetric kernel matrix.
        K = exp(-gamma * mat_sq_dists)
        # Center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        # Obtaining eigenpairs from the centered kernel matrix
        # numpy.eigh returns them in sorted order
        eigvals, eigvecs = eigh(K)
        # Collect the top k eigenvectors (projected samples)
        X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

        # Collect the corresponding eigenvalues
        lambdas = [eigvals[-i] for i in range(1, n_components + 1)]

        return X_pc, lambdas

    def project_x(self, x_new, X, gamma, alphas, lambdas):
        pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
        k = np.exp(-gamma * pair_dist)

        return k.dot(alphas / lambdas)
