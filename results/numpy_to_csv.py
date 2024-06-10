import numpy as np
import sys

def npy_to_csv(npy_file, csv_file):
    # Load the .npy file
    data = np.load(npy_file)
    if sys.argv[2] == '1': data = data[0, 0, :, :, 0, 0] 
    elif sys.argv[2] == '2': data = data[0, :, 0, :, 0]
    elif sys.argv[2] == '3': data = data[0, 0, :, :, 0]
    else: data = data[0, :, 0, :, 0, 0]
    print(data)

    # Open the CSV file for writing
    with open(csv_file, 'w') as f:
        # Write the header row (if desired)
        # f.write(','.join(['column{}'.format(i) for i in range(data.shape[1])]) + '\n')

        # Write the data rows
        for row in data:
            f.write(','.join(map(str, row)) + '\n')

# Example usage:
npy_file = sys.argv[1]
npy_to_csv(npy_file, npy_file.replace('.npy', '.csv'))
