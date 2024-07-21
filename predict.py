import pandas as pd
import sys
from LinearRegression import LinearRegression


def predict(theta0, theta1, mileage):
    """Wrapper of predict function of Linear Regression"""
    return LinearRegression._predict(theta0, theta1, mileage)


def main():
    try:
        df = pd.read_csv("theta.csv")
        theta0 = df["theta0"].values[0]
        theta1 = df["theta1"].values[0]
        mileage = 0
        for line in sys.stdin:
            try:
                mileage = float(line)
                break
            except Exception as e:
                print(f"error {type(e).__name__}  : {e}")
                return

        print(f"Predictions for {mileage} km  is  = {predict(theta0, theta1, mileage)}")
    except Exception as e:
        print(f"error {type(e).__name__}  : {e}")


if __name__ == "__main__":
    main()
