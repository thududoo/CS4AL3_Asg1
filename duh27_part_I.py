import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as an
import random
import time

CSV_PATH = "./gdp-vs-happiness.csv"
RATE = 1e-7
MAX_STEP = 1000000
THRESHOLD = 1e-16

class LRM:
    def __init__(self, csv_path, y_col, x_col):
        self.ys = []
        self.xs = []
        self.load_csv(csv_path, y_col, x_col)
        self.input = np.array(self.xs, dtype=float)
        self.target = np.array(self.ys, dtype=float)
        self.set_size = float(len(self.input))

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
        y_prid = slope * self.input + intrc
        d_intrc = sum(
                self.input * (self.target - y_prid)
            ) * -2 / self.set_size
        d_slope = sum(
            (self.target - y_prid)
            ) * -2 / self.set_size
        return d_slope, d_intrc

    def train_gd(self):
        slope = random.random()
        # intrc = random.random()
        # intrc = 0
        # slope = self.beta_ols[1]
        intrc = float(self.beta_ols[0])
        for _ in range(MAX_STEP):

            slope_pd, intrc_pd = self.gradient_loss_func(slope, intrc)
            # print(f"{slope=}, {intrc=}")
            # print(f"{slope_pd=}, {intrc_pd=}\n")
            if abs(slope_pd) <= THRESHOLD or abs(intrc_pd) <= THRESHOLD:
                print("break on threshold")
                break
            slope = slope - slope_pd * RATE
            intrc = intrc - intrc_pd * RATE * 5e-4

        print(f"{slope=}, {intrc=}")
        print(f"{slope_pd=}, {intrc_pd=}\n")
        xs = np.linspace(min(self.input), max(self.input), num=2)
        ys = slope * xs + intrc
        
        plt.plot(xs, ys)
        self.plot()


            
            
    
    def plot(self):
        plt.scatter(self.input, self.target)
        plt.show()




if __name__ == "__main__":
    model = LRM(csv_path=CSV_PATH, y_col=3, x_col=4)
    model.train_ols()
    print(model.beta_ols)
    model.train_gd()