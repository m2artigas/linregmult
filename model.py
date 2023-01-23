import numpy as np


class LinearRegression:

    def __init__(self, x_dim):
        self.theta = np.zeros(x_dim)
        self.cost_history = []
        self.theta_history = []

    def predict(self, X):
        """
        Computes the prediction (hzpothesis) of the linear regression
        :param X: input data as row vectors
        :return: vector of predicted outputs
        """
        return np.dot(self.theta, X.T)

    def cost(self, X, y):
        """
        Computes the loss function of a linear regression (mean square error)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Loss value
        """
        c = np.dot((np.dot(self.theta, X.T) - y).T, (np.dot(self.theta, X.T) - y))
        self.cost_history.append(c)
        return c
        # TODO

    def grad(self, X, y):
        """
        Computes the gradient of the loss function with regard to the parameters theta
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Gradient
        """
        gsol = []
        for i in range(len(self.theta)):
            aux = 0
            for j in range(X.shape[0]):
                aux += X[j][i] * (self.predict(X[j])) - y[j] * X[j][i]
            aux = aux/X.shape[0] 
            gsol.append(aux)
        return gsol

    def analytical_solution(self, X, y):
        """
        Computes analytical solution of the least-squares method (normal equation)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        aux = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)
        self.theta = np.dot(aux, y)
        
        self.theta_history.append(self.theta)
