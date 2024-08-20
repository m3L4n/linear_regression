import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing_data import denormalize_theta


class LinearRegression:

    def __init__(self, lr=0.01, n_iters=5_000):
        self.lr = lr
        self.n_iters = n_iters
        self.loss = []

    def cost_predict(self, theta0, theta1, mileage, price):
        """Compute predict - prices  ^ 2 to calcule the mean square error
        (calculate the distance between the real prices and the predict ( so the curve that fit b + ax))
        """
        return (self._predict(theta0, theta1, mileage) - price) ** 2

    def compute_cost(self, mileages, prices, theta0, theta1):
        """Compute the cost of gradient descent to see if the gradient descent learn well
        -> by calcule the error between prediction and real prices
        """
        m = mileages.shape[0]
        cost = 0
        for i in range(m):
            cost += self.cost_predict(theta0, theta1, mileages[i], prices[i])
        return (1 /  m) * cost

    def _compute_gradient(self, mileages, prices, theta0, theta1):
        """Compute gradient descent
        its an optimized algorithm that learn by itself how to fit with the data
        """
        th0_tmp = theta0
        th1_tmp = theta1
        m = len(mileages)
        predicts = [self._predict(theta0, theta1, mileage) for mileage in mileages]
        error = [predict - price for predict, price in zip(predicts, prices)]
        th0_tmp = (1 / m) * (np.sum(error))
        test = [x * y for x, y in zip(error, mileages)]

        th1_tmp = 1 / m * (np.sum(test))
        return th0_tmp, th1_tmp

    def fit(self, mileages, prices, mileages_Nnorm, prices_Nnorm):
        """Fit method of the class which calculate theta0 and theta1 with GD
        mileages and prices mus be standardized (z-scores algorithm)
        it save theta0 and theta1 in theta.csv at the root of the project
        """
        th0_tmp = 0.0
        th1_tmp = 0.0
        for _ in range(self.n_iters):
            dj_w, dj_db = self._compute_gradient(mileages, prices, th0_tmp, th1_tmp)
            th0_tmp = th0_tmp - self.lr * dj_w
            th1_tmp = th1_tmp - self.lr * dj_db
            self.loss.append(self.compute_cost(mileages, prices, th0_tmp, th1_tmp))
        t0_none_stand, t1_none_stand = denormalize_theta(
            th0_tmp,
            th1_tmp,
            mileages_Nnorm,
            prices_Nnorm,
        )

        self._save_theta(t0_none_stand, t1_none_stand)
        return t0_none_stand, t1_none_stand, th0_tmp, th1_tmp

    @staticmethod
    def _predict(theta0, theta1, mileage):
        """formula to predict the prices"""
        return theta0 + (theta1 * mileage)

    def print_cost(self):
        """Print the cost"""
        plt.plot(range(1, len(self.loss) + 1), self.loss)
        plt.title("Cost per iteration")
        plt.xlabel("Nbr iterations")
        plt.ylabel("Cost")
        plt.show()

    @staticmethod
    def _save_theta(theta0, theta1):
        """save theta0 and theta1 in theta csv"""
        data = [[theta0, theta1]]
        df = pd.DataFrame(data, columns=["theta0", "theta1"])
        df.to_csv(
            "theta.csv",
        )
