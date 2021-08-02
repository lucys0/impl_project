import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
import pandas as pd
import seaborn as sns
sns.set()

'''Read csv file '''


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))
    return x, y


# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'


plt.figure()
# x1, y1 = readcsv("plots/v1-oracle.csv")
# plt.plot(x1, y1, color='grey', label='oracle')

# x2, y2 = readcsv("plots/v1-cnn.csv")
# plt.plot(x2, y2, color='green', label='cnn')

# x3, y3 = readcsv("plots/v1-image-scratch.csv")
# plt.plot(x3, y3, color='blue', label='image_scratch')

# x4, y4 = readcsv("plots/v1-reward_prediction_finetune.csv")
# plt.plot(x4, y4, color='red', label='reward_prefiction_finetune')

data = pd.read_csv("plots/v1-cnn.csv")
sns.kdeplot(data=data, x="steps", y="values")

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # plt.xticks([0, 1e6, 2e6, 3e6, 4e6, 5e6])
# # plt.yticks([0, 10, 20, 30])

# # plt.ylim(22, 36)
# # plt.xlim(0, 5000000)
# plt.xlabel('Steps', fontsize=20)
# plt.ylabel('Value', fontsize=20)
# plt.legend(fontsize=16)
plt.show()
