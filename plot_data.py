import matplotlib.pyplot as plt
import numpy as np
from LinearRegression import LinearRegression


def plot_prediction(predictions, mileages, prices):
    """Plot predict to see if the line is inside the point"""
    plt.plot(mileages, prices, "ro")
    plt.plot(mileages, predictions, c="b", labels="Predict")
    plt.legend()
    plt.xlabel("X  mileage")
    plt.ylabel("Y  price")
    # plt.show()


def plot_repartiton(mileages, prices):
    """Plot repartitons of the data."""
    plt.plot(mileages, prices, "ro")
    plt.xlabel("X mileage")
    plt.ylabel("Y  price")
    # plt.show()


def score(predictions, prices):
    """R2 squared
    calculate if the model is performant
    formula  =  1 - sum(prices[i] - predict[i] )^2 / sum(prices[i] - mean(prices)^ 2)
    """
    numerator = np.sum(
        [(price - predict) ** 2 for predict, price in zip(predictions, prices)]
    )
    mean = np.mean(prices)
    denominator = np.sum([(price - mean) ** 2 for price in prices])
    formula = 1 - (numerator / denominator)
    print(f"Accuracy of the linear regression {formula}")


def plot_diff_stand(
    mileages,
    prices,
    mileages_stand,
    prices_stand,
    predicted_stand,
    predicted,
):
    """Plot on the same plot the prediction standardized and destandized
    its the same data but not on the same scale
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(mileages, prices, "bo")
    plt.plot(mileages, predicted, c="r")
    plt.title("non standardized data")
    plt.xlabel("X milage")
    plt.ylabel("Y price")

    plt.subplot(1, 2, 2)
    plt.plot(mileages_stand, prices_stand, "ro")
    plt.plot(mileages_stand, predicted_stand, c="b")
    plt.title("standardized data")
    plt.xlabel("X (standardized) mileage")
    plt.ylabel("Y (standardized) price")

    # plt.show()
