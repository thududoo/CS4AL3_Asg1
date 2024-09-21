import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as an
import random

CSV_PATH = "./training_data.csv"

class RM:
    def __init__(self) -> None:
        self.inputs = {}
        self.load_csv(CSV_PATH)

    def load_csv(self, csv_path):
        csv_data = pd.read_csv(csv_path)
        self.inputs["length"] = csv_data["Length"]
        self.inputs["diameter"] = csv_data["Diameter"]
        self.inputs["height"] = csv_data["Height"]
        self.inputs["whole_weight"] = csv_data["Whole_weight"]
        self.inputs["shucked_weight"] = csv_data["Shucked_weight"]
        self.inputs["viscera_weight"] = csv_data["Viscera_weight"]
        self.inputs["shell_weight"] = csv_data["Shell_weight"]
        self.target = np.array(csv_data["Rings"], dtype=float) + 1.5

    def train_ols(self):
        X = np.column_stack((np.ones(len(self.target)), *self.inputs.values()))
        Y = np.column_stack(self.target).T
        self.beta_ols = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def plot(self):
        fig, axs = plt.subplots(7)
        fig.suptitle("Age vs. Features")
        for ax, (key, val) in zip(axs, self.inputs.items()):
            ax.scatter(val, self.target, s=1)
            ax.set_title(key)
        plt.show()


if __name__ == "__main__":
    model = RM()
    model.train_ols()