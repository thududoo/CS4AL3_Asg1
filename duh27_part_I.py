import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as an
import random

CSV_PATH = "./gdp-vs-happiness.csv"
RATE = 1e-3
MAX_STEP = 300

class LRM:
    def __init__(self, csv_path):
        self.ys = []
        self.xs = []
        self.load_csv(csv_path)
        self.input = np.array(self.xs, dtype=float)
        self.target = np.array(self.ys, dtype=float)
        self.input = (self.input - np.mean(self.input)) / np.std(self.input)
        self.target = (self.target - np.mean(self.target)) / np.std(self.target)
        self.set_size = float(len(self.input))
        self.X = np.column_stack((np.ones(len(self.input)), self.input))
        self.Y = np.column_stack(self.target).T

    def load_csv(self, csv_path):
        # import data
        data = pd.read_csv(csv_path)
        #drop columns that will not be used
        by_year = (data[data['Year']==2018]).drop(columns=["Continent","Population (historical estimates)","Code"])
        # remove missing values from columns 
        df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2017 international $)']).notna()]
        for row in df.iterrows():
            if row[1]['Cantril ladder score']>4.5:
                self.ys.append(row[1]['Cantril ladder score'])
                self.xs.append(row[1]['GDP per capita, PPP (constant 2017 international $)'])

    def train_ols(self):
        X = self.X
        Y = self.Y
        self.beta_ols = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def train_gd(self):
        X = self.X
        Y = self.Y
        n = self.set_size
        self.beta_gd = {}
        for alpha in (1, 5):
            beta = np.random.randn(2, 1)
            for epoch in range(MAX_STEP):
                gradients =  2 / n * (X.T).dot(X.dot(beta) - Y)
                beta = beta - alpha * RATE * gradients
                if (epoch + 1) % (MAX_STEP / 3) == 0:
                    self.beta_gd[(alpha * RATE, epoch + 1)] = beta

    def loss(self, beta):
        y_pred = self.X.dot(beta)
        residuals = self.Y - y_pred
        return np.sum(residuals ** 2)

    def find_best_beta(self):
        min_RSE = float("inf")
        for re, beta in self.beta_gd.items():
            RSE = self.loss(beta)
            if RSE < min_RSE:
                min_RSE = RSE
                best_beta = beta
                best_rate_epoch = re
        return best_rate_epoch, best_beta

    def plot(self):
        xs = np.linspace(min(self.input), max(self.input), num=2)
        plt.figure()
        plt.xlabel("GDP")
        plt.ylabel("Happiness")
        plt.scatter(self.input, self.target, s=5)
        print(f"Gradient Descent Method:")
        for (rate, epoch), beta in self.beta_gd.items():
            ys = beta[1] * xs + beta[0]
            plt.plot(xs, ys,
                     label=f"GD@{rate=} & {epoch=}"
                     )
            print(f"\tbeta = [{float(beta[0][0]):.5E}, {float(beta[1][0]):.5E}]\t@ {rate = } & {epoch = }")
        plt.legend()
        print()
        plt.figure()
        plt.xlabel("GDP")
        plt.ylabel("Happiness")
        plt.scatter(self.input, self.target, s=5)
        ys = self.beta_ols[1] * xs + self.beta_ols[0]
        plt.plot(xs, ys, label="OLS", linewidth=3)
        (rate, epoch), beta = self.find_best_beta()
        ys = beta[1] * xs + beta[0]
        plt.plot(xs, ys,
                 label=f"Best GD@{rate=} & {epoch=}"
                 )
        print(f"Ordinary Least Squares Method:")
        print(f"\tbeta = [{float(self.beta_ols[0][0]):.5E}, {float(self.beta_ols[1][0]):.5E}]\t@ {rate = } & {epoch = }")
        print(f"Best from Gradient Descent Method:")
        print(f"\tbeta = [{float(beta[0][0]):.5E}, {float(beta[1][0]):.5E}]\t@ {rate = } & {epoch = }")
        plt.legend()
        plt.show()




if __name__ == "__main__":
    model = LRM(csv_path=CSV_PATH)
    model.train_ols()
    model.train_gd()
    model.plot()