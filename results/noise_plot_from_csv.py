import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import os

filepath = sys.argv[1]
data = []

if os.path.isfile(filepath):
    csvfile = open(filepath, 'r', newline='\n')

    reader = csv.reader(csvfile)
    data = [[float(n) if n != "" else 0 for n in row] for row in reader]

print(data)

fig, ax = plt.subplots()
#[[ax.bar(range(i*j), data[i][j], 0.5, label="{} {}".format(i, j)) for j in range(len(data[i]))] for i in range(len(data))]
[ax.plot(range(len(data[i])), data[i]) for i in range(len(data))]

plt.savefig(filepath.replace('.csv', '.jpg'))