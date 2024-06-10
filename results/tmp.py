import csv
import sys

# Specify the file paths
file1_path = sys.argv[1]
file2_path = sys.argv[2]
output_file_path = file1_path.replace('.csv', '_matched.csv')

# Read the data from the files
data1 = []
with open(file1_path, 'r') as file1:
    reader = csv.reader(file1)
    for row in reader:
        data1.append(row)

data2 = []
with open(file2_path, 'r') as file2:
    reader = csv.reader(file2)
    for row in reader:
        data2.append(row)

# Match the data series
matched_data = []
for i in range(len(data1)):
    matched_row = data1[i] + data2[i]
    matched_data.append(matched_row)

# Write the result to a new CSV file
with open(output_file_path, 'w+', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(matched_data)