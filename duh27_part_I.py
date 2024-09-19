import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as an
import random
import time

CSV_PATH = "./gdp-vs-happiness.csv"
RATE = 1e-2
MAX_STEP = 100
THRESHOLD = 1e-13

class LRM:
    def __init__(self, csv_path, y_col, x_col):
        self.ys = []
        self.xs = []
        self.load_csv(csv_path, y_col, x_col)
        self.input = np.array(self.xs, dtype=float)
        self.target = np.array(self.ys, dtype=float)

    def load_csv(self, csv_path, y_col, x_col):
        csv_file =  open(csv_path, "r")
        csv_file.readline()
        line = csv_file.readline()
        while line:
            line = line.split(",")
            if line[x_col] and line[y_col]:
                self.ys.append(line[y_col])
                self.xs.append(line[x_col])
            line = csv_file.readline()

    def train_ols(self):
        X = np.column_stack((np.ones(len(self.input)), self.input))
        Y = np.column_stack(self.target).T
        self.beta_ols = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def gradient_loss_func(self, slope, intrc):
        d_intrc = sum(
            -2 * (y - (intrc + slope * x))
            for x, y in zip(self.input, self.target)
            )
        d_slope = sum(
            -2 * x * (y - (intrc + slope * x))
            for x, y in zip(self.input, self.target)
            )
        return d_slope, d_intrc

    def train_gd(self):
        slope = random.random()
        intrc = random.random()
        for _ in range(MAX_STEP):
            slope_pd, intrc_pd = self.gradient_loss_func(slope, intrc)
            if abs(slope_pd) <= THRESHOLD or abs(intrc_pd) <= THRESHOLD:
                break
            slope -= slope_pd * RATE
            intrc -= intrc_pd * RATE
            print(f"{slope_pd=}, {intrc_pd=}")
            print(f"{slope=}, {intrc=}")
            xs = np.linspace(min(self.input), max(self.input), num=2)
            ys = slope * xs + intrc
            print(slope, intrc)
            plt.plot(xs, ys)
            self.plot()
            
    
    def plot(self):
        plt.scatter(self.input, self.target)
        plt.show()




if __name__ == "__main__":
    model = LRM(csv_path=CSV_PATH, y_col=3, x_col=4)
    model.train_gd()