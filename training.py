import pandas as pd

from LinearRegression import LinearRegression
from plot_data import plot_diff_stand, plot_repartiton, score

from preprocessing_data import prepocessing_data


def main():
    try:
        df = pd.read_csv("data.csv")
        mileages, prices, mileages_stand, prices_stand = prepocessing_data(dataFrame=df)

        lr = LinearRegression(lr=0.01, n_iters=500)
        (
            t0,
            t1,
            t0_stand,
            t1_stand,
        ) = lr.fit(mileages_stand, prices_stand, mileages, prices)

        # Bonus
        predicted_stand = [
            LinearRegression._predict(t0_stand, t1_stand, mileage)
            for mileage in mileages_stand
        ]
        predicted = [LinearRegression._predict(t0, t1, mileage) for mileage in mileages]
        lr.print_cost()
        plot_repartiton(mileages, prices)
        plot_diff_stand(
            mileages, prices, mileages_stand, prices_stand, predicted_stand, predicted
        )
        score(predicted, prices)
    except Exception as e:
        print(f"error {type(e).__name__}  : {e}")


if __name__ == "__main__":
    main()
