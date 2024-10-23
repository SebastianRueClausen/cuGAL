import json
import pandas as pd
import matplotlib.pyplot as plt

# Prompt user for JSON file path
json_file_path = input('Enter the path to your JSON file: ')

# Load JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame(data)

# Prompt user for CSV file path
csv_file_path = input('Enter the path to save the CSV file: ')
df.to_csv(csv_file_path, index=False)

# Prompt user for plot type
plot_type = input('Enter the type of plot (e.g., line, bar, scatter): ')

# Generate a plot
plt.figure(figsize=(10, 6))
df.plot(kind=plot_type)  # Use the user-specified plot type
plt.title('Your Plot Title')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# Prompt user for plot file path
plot_file_path = input('Enter the path to save the plot image: ')
plt.savefig(plot_file_path)