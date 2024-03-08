import sys
import csv


def read_from_csv(filename):
    data = {}
    with open(filename, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data["theta0"] = float(row["theta0"])
            data["theta1"] = float(row["theta1"])
    return data


def estimatePrice(mileage, t0, t1):
    predictedPrice = (t0 * float(mileage)) + t1
    return predictedPrice


def predictPrice(mileage):
    dict_parameter = read_from_csv("theta.csv")
    estimated_norm = estimatePrice(
        mileage, dict_parameter["theta0"], dict_parameter["theta1"]
    )
    return estimated_norm


def main(mileage):
    if not mileage.isnumeric():
        print("Error mileage must be numeric")
        exit()

    print(predictPrice(float(mileage)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error\nUsage python3 predictPrice.py [number mileage]")
    else:
        main(sys.argv[1])
