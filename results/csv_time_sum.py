# sum all columns in the csv file specifeid by the user and print the sum of each write to a new csv file
import csv
import sys
import numpy as np

file = sys.argv[1]
series = int(sys.argv[2])
with open(file, 'r') as f:
    reader = csv.reader(f)
    data = np.array(list(reader))

print(data)
# remove empty strings
data = data[:, data[0] != '']
print(data)
data = data.astype(np.float64)
print(data)
sums = np.sum(data, axis=0)
#stack the sums every n columns
sums = np.stack([sums[i:i+series] for i in range(0, len(sums), series)])
print(sums)

with open('sums.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(sums)
