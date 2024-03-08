from asyncio import constants
import csv
import numpy as np
import matplotlib.pyplot as plt
import copy

import pandas as pd

from predictPrice import estimatePrice, predictPrice


def load_data():
    df = pd.read_csv("data.csv")
    km = np.array(df["km"].values)
    price = np.array(df["price"].values)
    mean_km = np.mean(km)
    mean_price = np.mean(price)
    std_km = np.std(km)
    std_price = np.std(price)
    km_norm = (km - mean_km) / std_km

    min_price = np.min(price)
    max_price = np.max(price)
    price_norm = (price - min_price) / (max_price - min_price)
    return (
        km_norm,
        price_norm,
        mean_km,
        mean_price,
        std_km,
        std_price,
        max_price,
        min_price,
        km,
        price,
    )


def load_integral_data():
    df = pd.read_csv("data.csv")
    return np.array(df)


#  normalization required because the mileage had different echels, it change between + 1k to 10k but prices are more stable
#  so the mileage have to be normalize otherwise the result is inf or nan
#  normalization min - max
# data - min value / diff maxvalue  min value
def normalization_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    data_norm = (data - min_vals) / (max_vals - min_vals)
    return data_norm


def get_X_Y(data):
    return data[:, 0], data[:, 1]


def prediction(theta0, mileage, theta1):
    return theta0 * mileage + theta1


def denormalize_theta(theta0, theta1, min_price, max_price, min_mileage, max_mileage):
    denorm_theta0 = theta0 * (max_price - min_price) / (max_mileage - min_mileage)
    denorm_theta1 = (
        theta1 * (max_price - min_price) - denorm_theta0 * min_mileage + min_price
    )
    return denorm_theta0, denorm_theta1


def cost_predict(theta0, theta1, mileage, price):
    return (prediction(theta0, mileage, theta1) - price) ** 2


def compute_cost(mileages, prices, theta0, theta1):
    m = mileages.shape[0]
    cost = 0
    for i in range(m):
        cost += cost_predict(theta0, theta1, mileages[i], prices[i])
    return (1 / (2 * m)) * cost


def compute_gradient(mileages, prices, theta0, theta1):
    m = mileages.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        predict = prediction(theta0, mileages[i], theta1)
        dj_db += predict - prices[i]
        dj_dw += (predict - prices[i]) * mileages[i]
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradent_descent(mileages, prices, theta0, theta1, learning_rate, num_iters):
    w = copy.deepcopy(theta0)
    b = theta1
    history_cost = []
    for idx in range(num_iters):
        dj_dw, dj_db = compute_gradient(mileages, prices, w, b)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        history_cost.append(
            {
                "theta0": w,
                "theta1": b,
                "cost": compute_cost(mileages, prices, w, b),
                "iteration": idx,
            }
        )

    return w, b, history_cost


def write_to_csv(
    filename, theta0, theta1, min_price, max_price, max_mileage, min_mileage
):
    data = {
        "theta0": theta0,
        "theta1": theta1,
        "min_price": min_price,
        "max_price": max_price,
        "min_mileage": min_mileage,
        "max_mileage": max_mileage,
    }
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)


def show_cost(
    history_cost,
):
    history_sorted = sorted(history_cost, key=lambda k: k["cost"])
    cost = [d["cost"] for d in history_cost]
    plt.plot(range(1, len(cost) + 1), cost)
    plt.title("Cost per iteration")
    plt.xlabel("Nbr iterations")
    plt.ylabel("Cost")
    plt.show()


def main():
    data = load_integral_data()
    mileages, prices = get_X_Y(data)
    data_normalized = normalization_data(data)
    mileages_norm, prices_norm = get_X_Y(data_normalized)
    initial_t0 = 0.0
    initial_t1 = 0.0
    iterations = 7000
    learning_rate = 0.01
    w, b, history_cost = gradent_descent(
        mileages_norm,
        prices_norm,
        initial_t0,
        initial_t1,
        learning_rate,
        iterations,
    )
    m = mileages_norm.shape[0]
    predicted = np.zeros(m)
    predicted_non_norm = np.zeros(m)
    min_price = np.min(
        prices,
    )
    max_price = np.max(prices)
    min_mileage = np.min(mileages)
    max_mileage = np.max(mileages)
    theta0_non_norm, theta1_non_norm = denormalize_theta(
        w, b, min_price, max_price, min_mileage, max_mileage
    )
    write_to_csv(
        "theta.csv",
        theta0_non_norm,
        theta1_non_norm,
        min_price,
        max_price,
        min_mileage,
        max_mileage,
    )
    show_cost(history_cost)
    for i in range(m):
        predicted[i] = prediction(w, mileages_norm[i], b)
    for i in range(m):
        predicted_non_norm[i] = prediction(
            theta0_non_norm, mileages[i], theta1_non_norm
        )

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(mileages, prices, "bo")
    plt.plot(mileages, predicted_non_norm, c="r")
    plt.title("non noramlized data")
    plt.xlabel("X milage")
    plt.ylabel("Y price")

    plt.subplot(1, 2, 2)
    plt.plot(mileages_norm, prices_norm, "ro")
    plt.plot(mileages_norm, predicted, c="b")
    plt.title("normalized data")
    plt.xlabel("X (normalisé) mileage")
    plt.ylabel("Y (normalisé) price")

    plt.show()
    plt.tight_layout()


if __name__ == "__main__":
    main()
