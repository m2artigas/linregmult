import numpy as np


class Optimizer:

    def __init__(self, model):
        self.model = model
        self.iter = 0

    def step(self, X, y):
        """
        Performs a single step of the gradient descent
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        raise NotImplementedError("Method not yet implemented.")

    def converged(self):
        """

        :return: True if the gradient descent iteration ended
        """
        raise NotImplementedError("Method not yet implemented.")

    def optimize_full_batch(self, X, y):
        """
        Runs the optimization processing all the data at each step
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        while not self.converged():
            self.step(X, y)
            self.iter += 1


class GradientDescent(Optimizer):

    def __init__(self, model, alpha=0.3, num_iters=1000, min_cost=0, min_theta_diff=0, **options):
        super(GradientDescent, self).__init__(model)
        self.options = options
        self.alpha = alpha
        self.num_iters = num_iters
        self.min_cost = min_cost
        self.min_theta_diff = min_theta_diff
        self.cost = np.Inf

    def step(self, X, y):
        """
        Performs a single step of the gradient descent
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """

        gr = self.model.grad(X, y)
        aux = []
        for j in range(len(self.model.theta)):
             aux.append(self.model.theta[j] - self.alpha * gr[j])
        self.model.theta = aux
        self.model.theta_history.append(self.model.theta)
        self.cost = self.model.cost(X, y)
        return None

    def converged(self):
        """

        :return: True if the gradient descent iteration ended
        """
        if self.num_iters <= self.iter or self.cost <= self.min_cost:
            return True
        if len(self.model.theta_history) > 1:
            if np.allclose(self.model.theta_history[-1], self.model.theta_history[-2], self.min_theta_diff): return True
        else:
            return False


