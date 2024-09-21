import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as an
import random

CSV_PATH = "./gdp-vs-happiness.csv"
# Learning rate for gradient descent
RATE = 1e-3
# Maximum number of iterations for gradient descent
MAX_STEP = 300


class LRM:
    """
    Linear Regression Model class to perform Ordinary Least Squares (OLS)
    and Gradient Descent (GD) on a dataset for predicting the relation between
    happiness vs. GDP.
    """

    def __init__(self, csv_path: str) -> None:
        """
        Initialize the LRM object by loading the dataset and preparing input
        and target data for regression.

        Args:
            csv_path (str): The file path to the CSV file containing the dataset.
        """

        self.ys = []
        self.xs = []
        self.load_csv(csv_path)
        self.input = np.array(self.xs, dtype=float)
        self.target = np.array(self.ys, dtype=float)

        # Normalize input and target data
        self.input = (self.input - np.mean(self.input)) / np.std(self.input)
        self.target = (self.target - np.mean(self.target)) / np.std(self.target)
        # Calculate the data set size
        self.set_size = float(len(self.input))

        # Compose X and Y matrices
        self.X = np.column_stack((np.ones(len(self.input)), self.input))
        self.Y = np.column_stack(self.target).T

    def load_csv(self, csv_path: str) -> None:
        """
        Load the dataset from the given CSV file, filter rows for the year 2018,
        and extract GDP and happiness scores.

        Args:
            csv_path (str): The file path to the CSV file containing the dataset.
        """

        # Import data
        data = pd.read_csv(csv_path)
        # Drop columns that will not be used
        by_year = (data[data["Year"] == 2018]).drop(
            columns=["Continent", "Population (historical estimates)", "Code"]
        )
        # Remove missing values from columns
        df = by_year[
            (by_year["Cantril ladder score"].notna())
            & (by_year["GDP per capita, PPP (constant 2017 international $)"]).notna()
        ]

        # Append data to xs and ys lists
        for row in df.iterrows():
            if row[1]["Cantril ladder score"] > 4.5:
                self.ys.append(row[1]["Cantril ladder score"])
                self.xs.append(
                    row[1]["GDP per capita, PPP (constant 2017 international $)"]
                )

    def train_ols(self) -> None:
        """
        Train the model using Ordinary Least Squares (OLS) method and
        calculate the regression coefficients (beta_ols).
        """

        X = self.X
        Y = self.Y
        # Calculate the OLS beta coefficients
        self.beta_ols = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def train_gd(self) -> None:
        """
        Train the model using Gradient Descent (GD) and compute regression
        coefficients for different learning rates (alpha) and epochs.
        """

        X = self.X
        Y = self.Y
        n = self.set_size
        self.beta_gd = {}  # Store betas for different epochs and learning rates

        # Loop over different learning rate scalers (alpha)
        for alpha in (1, 5):
            beta = np.random.randn(2, 1)  # Initialize random beta values
            # Gradient descent loop
            for epoch in range(MAX_STEP):
                # Compute gradients
                gradients = 2 / n * (X.T).dot(X.dot(beta) - Y)
                # Update beta values based on gradients
                beta = beta - alpha * RATE * gradients
                # Save beta value 3 times along the process
                if (epoch + 1) % (MAX_STEP / 3) == 0:
                    self.beta_gd[(alpha * RATE, epoch + 1)] = beta

    def loss(self, beta: np.ndarray) -> float:
        """
        Calculate the residual sum of squares (RSS) loss for a given set of
        regression coefficients.

        Args:
            beta (np.ndarray): The regression coefficients for the model.

        Returns:
            float: The residual sum of squares (RSS).
        """

        y_pred = self.X.dot(beta)
        residuals = self.Y - y_pred
        return np.sum(residuals**2)

    def find_best_beta(self) -> tuple:
        """
        Find the best set of regression coefficients (beta) from the Gradient
        Descent training by minimizing the loss function.

        Returns:
            Tuple[Tuple[float, int], np.ndarray]:
                A tuple where the first element is a tuple containing
                the learning rate and epoch (both floats), and the second
                element is the corresponding beta values (np.ndarray).
        """

        min_RSS = float("inf")  # Initialize with a very large number
        # Iterate over all stored beta values
        for re, beta in self.beta_gd.items():
            RSS = self.loss(beta)  # Calculate RSS
            if RSS < min_RSS:
                min_RSE = RSS
                best_beta = beta
                best_rate_epoch = re
        return best_rate_epoch, best_beta

    def plot(self) -> None:
        """
        Plot the regression results comparing the Ordinary Least Squares (OLS)
        and Gradient Descent (GD) methods.

        Displays scatter plots of the input data and the regression lines.
        """

        # Create line for plotting regression results
        xs = np.linspace(min(self.input), max(self.input), num=2)

        # Plot the data and gradient descent results
        plt.figure()
        plt.xlabel("GDP")
        plt.ylabel("Happiness")
        # Scatter plot of the data points
        plt.scatter(self.input, self.target, s=5)

        print(f"Gradient Descent Method:")
        # Iterate over all stored beta values
        for (rate, epoch), beta in self.beta_gd.items():
            ys = beta[1] * xs + beta[0]
            plt.plot(xs, ys, label=f"GD@{rate=} & {epoch=}")
            print(
                f"\tbeta = [{float(beta[0][0]):.5E}, {float(beta[1][0]):.5E}]\t@ {rate = } & {epoch = }"
            )
        plt.legend()
        print()  # Add an empty line

        # Plot OLS results and the best gradient descent result
        plt.figure()
        plt.xlabel("GDP")
        plt.ylabel("Happiness")
        # Scatter plot of the data points
        plt.scatter(self.input, self.target, s=5)
        ys = self.beta_ols[1] * xs + self.beta_ols[0]
        plt.plot(xs, ys, label="OLS", linewidth=3)  # Plot OLS line

        # Get best GD result
        (rate, epoch), beta = self.find_best_beta()
        ys = beta[1] * xs + beta[0]
        plt.plot(xs, ys, label=f"Best GD@{rate=} & {epoch=}")  # Plot Best GD line

        print(f"Ordinary Least Squares Method:")
        print(
            f"\tbeta = [{float(self.beta_ols[0][0]):.5E}, {float(self.beta_ols[1][0]):.5E}]\t@ {rate = } & {epoch = }"
        )
        print(f"Best from Gradient Descent Method:")
        print(
            f"\tbeta = [{float(beta[0][0]):.5E}, {float(beta[1][0]):.5E}]\t@ {rate = } & {epoch = }"
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    model = LRM(csv_path=CSV_PATH)
    model.train_ols()
    model.train_gd()
    model.plot()
