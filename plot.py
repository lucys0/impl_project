import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# def readcsv(files):
#     csvfile = open(files, 'r')
#     plots = csv.reader(csvfile, delimiter=',')
#     x = []
#     y = []
#     for row in plots:
#         y.append((row[2]))
#         x.append((row[1]))
#     return x, y


def plot(file_path, label, smoothing_coef=0.00115):
    df = pd.read_csv(file_path)

    # Determine smoothing factor from rule of thumb (use f = 0.05)
    smooth_factor = smoothing_coef * (df["Value"] ** 2).sum()

    # Set up an scipy.interpolate.UnivariateSpline instance
    x = df["Step"].values
    y = df["Value"].values
    spl = UnivariateSpline(x, y, s=smooth_factor)

    # spl is now a callable function
    x_spline = np.linspace(x[0], x[-1], 400)
    y_spline = spl(x_spline)

    # Plot results
    plt.plot(x_spline, y_spline, label=label, lw=2)

plt.figure(figsize=(11, 6))
with plt.style.context('seaborn-darkgrid'):
    plot("plots/v0-oracle.csv", label='oracle', smoothing_coef=0.0003)
    plot("plots/v0-image_scratch.csv", label='image_scratch', smoothing_coef=0.00135)
    plot("plots/v0-cnn.csv", label='cnn', smoothing_coef=0.0014)
    plot("plots/v0-reward_prediction_finetune.csv", label='reward_prefiction_finetune', smoothing_coef=0.004)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(fontsize=14)
    plt.title('Follow (0 distractor)')

plt.show()
