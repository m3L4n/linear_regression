import numpy as np


def normalize_data(X):
    """X -> Ndarray Normalized data  with z-index algorithmn
    because Its need to be on the same scale"""
    data_norm = (X - np.mean(X)) / np.std(X)
    return data_norm


def denormalize_theta(theta0, theta1, mileages, prices):
    """un standardize theta
    Be careful : you must use mileages none standardized and prices none standardized
    formula : new_theta1 = theta1 * ecart_type y / ecart type x
    new theta0 = mean y - theta1 * mean x
    """
    std_y = np.std(prices)
    std_x = np.std(
        mileages,
    )
    mean_y = np.mean(prices)
    mean_x = np.mean(mileages)

    t1 = theta1 * (std_y / std_x)
    t0 = mean_y - (t1 * mean_x)

    return t0, t1


def prepocessing_data(dataFrame):
    """Get mileages and prices from dataframe
    return standardized prices and mileages
    """
    try:
        mileages = np.array(dataFrame["km"].values)
        prices = np.array(dataFrame["price"].values)
        mileages_stand = normalize_data(mileages)
        prices_stand = normalize_data(prices)
        return mileages, prices, mileages_stand, prices_stand
    except Exception:
        return [], [], [], []
