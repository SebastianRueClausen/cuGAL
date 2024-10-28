import json
import re
import matplotlib.pyplot as plt
import readline
import os
import numpy as np
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
file = open(json_file_path, 'r')
results = ExperimentResults.from_dict(json.load(file))

num_noises  = len(results.experiment.noise_levels)
num_results = len(list(results.all_results()))
num_graphs  = len(results.experiment.graphs)
num_algs    = len(results.experiment.algorithms)
print("num_noises ", num_noises, " num_results ", num_results, " num_graphs ", num_graphs, " num_algs ", num_algs)

# Bar plot the accuracy with labels for each experiment
cmap = plt.get_cmap('tab20')

alg_desc = ExperimentResults.algorithms_descriptions(results.experiment.algorithms)

def legend_without_duplicate_labels(labels):
    unique = [l for i, l in enumerate(labels) if l not in labels[:i]]
    plt.legend(unique, loc='center left', bbox_to_anchor=(1, 0.5))

def draw_x_axis_labels():
    x_pos = np.linspace(num_results / num_noises / 2, num_results - (num_results / num_noises / 2), num_noises)
    plt.xticks(x_pos, [str(n.source_noise*100) + "% Noise" for n in results.experiment.noise_levels])

    ## Draw vertical lines (I am not including the lines at the ends)
    pos = [((i)*num_algs-0.5) for i in range(num_graphs+1)]
#    plt.vlines(pos[1:-1], 0, -0.4, color='black', lw=1.5, clip_on=False, transform=plt.gca().get_xaxis_transform())

    ## Draw second level axes ticklables
    for ps0, ps1, lbl in zip(pos[:-1], pos[1:], [re.sub(" {|, ", "\n{", str(g)) for g in results.experiment.graphs]):
        plt.text((ps0 + ps1) / 2, -0.16, lbl, ha='center', clip_on=False, transform=plt.gca().get_xaxis_transform(), weight = 'bold', size=6)
    for i, res in enumerate(results.all_results()):
        plt.bar(i, res[3].accuracy, label=alg_desc[res[2]],
                color=cmap(int.from_bytes(str(res[2]).encode(), 'little') % 20))

## Draw the Accuracy plot
draw_x_axis_labels()
legend_without_duplicate_labels(plt.gca().get_legend_handles_labels()[1])


plt.ylabel('Accuracy')
plt.title('Accuracy of Experiments')
plt.savefig(file.name + '_accuracy.png', dpi=300, bbox_inches = "tight")

# Draw the Time plot
plt.clf()
for i, res in enumerate(results.all_results()):
    plt.bar(i, res[3].profile.time, label=alg_desc[res[2]], 
            color=cmap(int.from_bytes(str(res[2]).encode(), 'little') % 20))
    
draw_x_axis_labels()
legend_without_duplicate_labels(plt.gca().get_legend_handles_labels()[1])

plt.ylabel('Time (s)')
plt.title('Time of Experiments')
plt.savefig(file.name + '_time.png', dpi=300, bbox_inches = "tight")