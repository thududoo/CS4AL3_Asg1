import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as an
import random

CSV_PATH = "./training_data.csv"
r = 1  # Train to test ratio


class RM:
    def __init__(self) -> None:
        self.inputs = {}
        self.all_data = {}
        self.load_csv(CSV_PATH)

    def load_csv(self, csv_path: str) -> None:
        """
        Loads data from a CSV file, shuffles it, and splits it into training
        and testing sets based on the specified ratio.

        Args:
            csv_path (str): Path to the CSV file containing the data.
        """

        csv_data = pd.read_csv(csv_path)

        # Calculate the split index for r to 1 train to test ratio
        split_index = int(len(csv_data) * (r / (r + 1)))

        # Shuffle the data and split into train and test sets
        csv_data = csv_data.sample(frac=1)
        # Split the data
        self.train_data = csv_data[:split_index]
        self.test_data = csv_data[split_index:]

        # Load input data (all features except the first and last columns)
        for attr in csv_data.columns.to_list()[1:-1]:
            self.all_data[attr] = csv_data[attr]
            self.inputs[attr] = self.train_data[attr]

        # Store target values (Rings + 1.5 as the age offset)
        self.target = np.array(self.train_data["Rings"], dtype=float) + 1.5
        self.all_targ = np.array(csv_data["Rings"], dtype=float) + 1.5

    def train_ols(self) -> None:
        """
        Trains an Ordinary Least Squares (OLS) regression model using the
        training data.
        From the scatter plot it is obvious that the age to Length, Diameter,
        Height are linear and rest are second ordered polynomial.
        Creates the feature matrix X and target vector Y, and calculates
        the OLS coefficients (beta_ols).
        """

        # Construct the feature matrix X with a bias term (1) and all input features
        X = np.column_stack(
            (
                np.ones(len(self.target)),
                self.inputs["Length"],
                self.inputs["Diameter"],
                self.inputs["Height"],
                self.inputs["Whole_weight"], self.inputs["Whole_weight"] ** 2,
                self.inputs["Shucked_weight"], self.inputs["Shucked_weight"] ** 2,
                self.inputs["Viscera_weight"], self.inputs["Viscera_weight"] ** 2,
                self.inputs["Shell_weight"], self.inputs["Shell_weight"] ** 2,
            )
        )
        Y = np.column_stack(self.target).T

        # Calculate beta (OLS coefficients) using the normal equation
        self.beta_ols = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(
        self,
        length: float,
        diameter: float,
        height: float,
        whole_weight: float,
        shucked_weight: float,
        viscera_weight: float,
        shell_weight: float,
    ) -> float:
        """
        Predicts the target value (age) using the trained OLS model
        and given feature values.

        Args:
            length (float): Length of the abalone.
            diameter (float): Diameter of the abalone.
            height (float): Height of the abalone.
            whole_weight (float): Whole weight of the abalone.
            shucked_weight (float): Shucked weight of the abalone.
            viscera_weight (float): Viscera weight of the abalone.
            shell_weight (float): Shell weight of the abalone.

        Returns:
            float: Predicted age based on the given features.
        """

        # Create a feature vector from the input parameters
        x = np.array(
            (
                1,
                length,
                diameter,
                height,
                whole_weight, whole_weight**2,
                shucked_weight, shucked_weight**2,
                viscera_weight, viscera_weight**2,
                shell_weight, shell_weight**2,
            )
        )
        # Predict target value by applying the OLS model (beta coefficients)
        y = x.dot(self.beta_ols)
        return float(y[0])

    def evaluate(self) -> float:
        """
        Evaluates the model on the test dataset and calculates the
        Mean Absolute Error (MAE).

        Returns:
            float: Mean Absolute Error of the predictions on the test data.
        """

        n = float(len(self.test_data))  # Total number of test data points
        sum_error = 0

        # Iterate over each test data sample
        for i in self.test_data.itertuples():
            paras = i[2:-1]  # Extract feature values
            target_age = i[-1] + 1.5  # Actual target age
            predict_age = self.predict(*paras)  # Predicted age from model
            error = abs(target_age - predict_age)
            sum_error += error  # Accumulate the errors

            # Output a comparison list of actual and predicted age
            # print(
            #     f"Target age: {target_age},\t"
            #     + f"predicted age: {predict_age:.2f},\t"
            #     + f"error: {error:.2f}\n"
            # )

        # Compute Mean Absolute Error (MAE)
        MAE = sum_error / n
        return MAE

    def plot(self) -> None:
        """
        Creates scatter plots for each feature against the actual and
        predicted ages.
        The red dots represent predicted ages, while the blue dots represent
        actual ages.
        """

        # Create a 4 by 2 plot window for all features
        fig, axs = plt.subplots(2, 4)
        fig.suptitle("Age vs. Features")

        predicted_ages = []
        # Iterate over all data points (using the index of one feature, such as Length)
        for i in self.all_data["Length"].index:
            paras = [
                self.all_data[attr][i] for attr in self.inputs.keys()
            ]  # Fetch all feature values for row i
            predicted_age = self.predict(*paras)
            predicted_ages.append(predicted_age)

        # Plot the scatter of actual values and the predicted line
        for i, (key, val) in enumerate(self.all_data.items()):
            axs[i // 4, i % 4].scatter(val, self.all_targ, s=1, label="Actual")
            axs[i // 4, i % 4].scatter(
                val, predicted_ages, s=1, color="red", label="Predicted Age"
            )
            axs[i // 4, i % 4].set_title(f"Age vs. {key}")
            axs[i // 4, i % 4].legend()

        plt.show()


if __name__ == "__main__":
    model = RM()
    model.train_ols()
    MAE = model.evaluate()
    print("Model calculated beta:")
    print(model.beta_ols)
    print(f"Model prediction MAE = {MAE}")
    model.plot()
