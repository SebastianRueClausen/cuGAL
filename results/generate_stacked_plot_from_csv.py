import matplotlib.pyplot as plt
import csv
import sys
import os
import numpy as np

filepath = sys.argv[1]
option = sys.argv[2]
data_labels = sys.argv[3:]
data = []
sizes = [128, 256, 512, 1024]

if os.path.isfile(filepath):
    csvfile = open(filepath, 'r', newline='')

    reader = csv.reader(csvfile)
    data = [[float(n) if n != "" else 0 for n in row][1:] for row in reader]
print(data)
n = len(data[0])

index = None
labels = ["Sinkhorn-Knopp", "Feature Extraction", "Gradient", "Hungarian"]
#labels = [None]*n
if option != "":
    index = labels.index(option)
    labels = labels[index]
#plt.stackplot(
#    sizes,
#    data,
#    labels=labels,
#    baseline ='zero'
#)


fig, ax = plt.subplots()
bottom = np.zeros(n)

print(len(data))
if index is None:
    for i in range(len(data)):
        print(i)
        print(data[:][i])
        p = ax.bar(range(n), data[:][i], 0.5, label=labels[i], bottom=bottom)
        bottom += data[i]
else:
    ax.bar(range(n), data[:][index], 0.5, label=labels)

#ax.set_yscale('log')
#plt.ylim(0, 270)

#plt.xticks(range(n), ["$2^{10}$", "$2^{11}$", "$2^{12}$", "$2^{13}$", "$2^{14}$"]*3)
plt.xlabel("Graph size")
plt.ylabel("Time (seconds)")
plt.legend()
plt.savefig("timeplot.jpg")
plt.show()
