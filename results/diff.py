# Load the specified csv file and subtract all values by the value of the first row.

import csv
import sys
import os
import numpy as np

filepath = sys.argv[1]
data = []

if os.path.isfile(filepath):
    csvfile = open(filepath, 'r', newline='')

    reader = csv.reader(csvfile)
    data = np.array([[float(n) if n != "" else 0 for n in row] for row in reader])


data = (data.T - data.T[0]).T

print(data)

#Write the result to a new CSV file without the first row
with open(filepath.replace('.csv', '_normalized.csv'), 'w+', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(data[1:])