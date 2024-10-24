import json
import matplotlib.pyplot as plt
import readline
import os
from experiment import ExperimentResults

def complete_path(text, state):
    # Split the text into directory and file parts
    dir_path, file_part = os.path.split(text)
    #print("dir_path ", dir_path, "file_part ", file_part, "state ", state, "text ", text, "\n")
    
    # Default to current directory if no directory is specified
    if not dir_path:
        dir_path = '.'
    
    # List all files in the directory and filter by the file part
    try:
        matches = [x + ('/' if os.path.isdir(os.path.join(dir_path, x)) else '') for x in os.listdir(dir_path) if x.startswith(file_part)]
    except FileNotFoundError:
        matches = []

    #print("matches ", len(matches), " ", matches, "\n")

    #if len(matches) == 1:
    #    matches[0] = dir_path + "/" + matches[0]
    # Return the state'th match
    try:
        return dir_path + "/" + matches[state]
    except IndexError:
        return None

# Set the completer function
readline.set_completer(complete_path)

# Adjust the delimiters to include '/'
readline.set_completer_delims(readline.get_completer_delims().replace('/', ''))
readline.set_completer_delims(readline.get_completer_delims().replace('-', ''))
readline.set_completer_delims(readline.get_completer_delims().replace(':', ''))

# Enable tab completion
readline.parse_and_bind("tab: complete")

# Prompt user for JSON file path with tab completion
while True:
    json_file_path = input('Enter the path to the JSON file: ')
    if os.path.isfile(json_file_path):
        break
    else:
        print('File not found, try again.')

# Load JSON data
with open(json_file_path, 'r') as file:
    results = ExperimentResults.from_dict(json.load(file))

# Bar plot the accuracy with labels for each experiment
cmap = plt.get_cmap('tab20')

alg_desc = ExperimentResults.algorithms_descriptions(results.experiment.algorithms)

def legend_without_duplicate_labels(labels):
    unique = [l for i, l in enumerate(labels) if l not in labels[:i]]
    plt.legend(unique, loc='center left', bbox_to_anchor=(1, 0.5))

for i, res in enumerate(results.all_results()):
    plt.bar(i, res[3].accuracy, label=alg_desc[res[2]],
            color=cmap(int.from_bytes(str(res[2]).encode(), 'little') % 20))

legend_without_duplicate_labels(plt.gca().get_legend_handles_labels()[1])

plt.xlabel('Experiment')
plt.ylabel('Accuracy')
plt.title('Accuracy of Experiments')
plt.savefig('accuracy.png', dpi=300, bbox_inches = "tight")

# Bar plot the time with labels for each experiment
plt.clf()
for i, res in enumerate(results.all_results()):
    plt.bar(i, res[3].profile.time, label=alg_desc[res[2]], 
            color=cmap(int.from_bytes(str(res[2]).encode(), 'little') % 20))
    
legend_without_duplicate_labels(plt.gca().get_legend_handles_labels()[1])

plt.legend()
plt.xlabel('Experiment')
plt.ylabel('Time (s)')
plt.title('Time of Experiments')
plt.savefig('time.png', dpi=300, bbox_inches = "tight")
