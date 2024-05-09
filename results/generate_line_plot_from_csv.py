import matplotlib.pyplot as plt
import csv
import sys
import os
import numpy as np

filepath = sys.argv[1]
data = []
sizes = [1024, 2048, 4096, 8192]

if os.path.isfile(filepath):
    csvfile = open(filepath, 'r', newline='')

    reader = csv.reader(csvfile)
    data = [[float(n) if n != "" else 0 for n in row][1:] for row in reader]
print(data)

n = len(data[0])

fig, ax = plt.subplots()
bottom = np.zeros(n)

print(len(data))

print(data[0][::2], data[2][1::2])
p = ax.plot(range(4), data[2][::2], 0.5)
p = ax.plot(range(4), data[2][1::2], 0.5)

plt.xticks(range(4), sizes)
plt.xlabel("Graph size")
plt.ylabel("Time (seconds)")
plt.legend()
plt.savefig("timeplot.jpg")
plt.show()
