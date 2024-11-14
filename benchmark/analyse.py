from dataclasses import dataclass
import json
import re
import matplotlib.pyplot as plt
import readline
import os
import numpy as np
import copy
from enum import Enum
from experiment import Algorithm, ExperimentResults, Result, Graph, GraphKind, Experiment

class PlotType(Enum):
    BAR = "bar"
    HEATMAP = "heatmap"

@dataclass
class Plot():
    plot_type: PlotType
    params: dict

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

def setup_readline():
    # Set the completer function
    readline.set_completer(complete_path)

    # Adjust the delimiters to include '/'
    readline.set_completer_delims(readline.get_completer_delims().replace('/', ''))
    readline.set_completer_delims(readline.get_completer_delims().replace('-', ''))
    readline.set_completer_delims(readline.get_completer_delims().replace(':', ''))

    # Enable tab completion
    readline.parse_and_bind("tab: complete")

def prompt_user_and_load_results() -> str:
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
    return results, file

def prompt_user_for_analysis_options(results: ExperimentResults) -> tuple[ExperimentResults, PlotType]:
    # Ask the user if all graphs should be averaged
    average_graphs = input('Do you want to average the results for all graphs? (y/[n]): ')
    if average_graphs.lower() == 'y':
        # Average the results for all graphs
        averaged_results = []
        for noise_lvl in results.experiment.noise_levels:
            for alg in results.experiment.algorithms:
                graph_results = filter(lambda r: r[1] == noise_lvl and r[2] == alg, results.all_results())            
                averaged_results.append(
                    Result.average([r[3] for r in graph_results]))
        combined_experiment = copy.deepcopy(results.experiment)
        combined_experiment.graphs = graphs=[Graph(GraphKind.AVERAGED, 
                                                   {'graphs': results.experiment.graphs})]
        results = copy.deepcopy(results)
        results.experiment = combined_experiment
    
    # Ask the user how to plot the results
    type_prompt = input('Do you want to plot the results as a bar plot or a heatmap? ([[b]ar]/[h]eatmap): ')
    if type_prompt.lower() == 'h':
        plot_type = PlotType.HEATMAP

        # Prompt the user for what to plot on each axis
        x_axis = input('What do you want to plot on the x-axis? ([g]raphs/[a]lgorithms/[n]oise levels/[c]ustom): ')
        y_axis = input('What do you want to plot on the y-axis? ([g]raphs/[a]lgorithms/[n]oise levels/[c]ustom): ')

        plot = Plot(plot_type, {'x_axis': x_axis.lower(), 'y_axis': y_axis.lower()})

    else:
        plot_type = PlotType.BAR
        plot = Plot(plot_type, {})

    return results, plot

def bar_plot_results(results: ExperimentResults, plot: Plot):
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
        for ps0, ps1, lbl in zip(pos[:-1], pos[1:], [ExperimentResults.graph_descriptions(results.experiment.graphs)[str(g)] for g in results.experiment.graphs]): #[re.sub(" {|, ", "\n{", str(g)) for g in results.experiment.graphs]):
            plt.text((ps0 + ps1) / 2, -0.16, lbl, ha='center', clip_on=False, transform=plt.gca().get_xaxis_transform(), weight = 'bold', size=6)
        for i, res in enumerate(results.all_results()):
            plt.bar(i, res[3].accuracy, label=alg_desc[res[2]],
                    color=cmap(int.from_bytes(str(res[2]).encode(), 'little') % 20))

    ## Draw the Accuracy plot
    draw_x_axis_labels()
    print("Legend handles labels ", plt.gca().get_legend_handles_labels())
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

def prompt_custom_heatmap_axis(results: ExperimentResults):
    data_choice = input('Select from [g]raphs, [a]lgorithms, [n]oise levels: ')
    match data_choice: 
        case 'g': data = results.experiment.graphs
        case 'a': data = results.experiment.algorithms
        case 'n': data = results.experiment.noise_levels
        case _: raise ValueError('Invalid data choice')        
        
    group_or_series = input('Would you like to select [s]pecific series or [g]roup series: ')
    if group_or_series == 's':
        series_range = input('Enter the series you would like to group by: ')
        series = np.array([eval(f"data[{series_range}])") for i in series_range.split(',')]).flatten()
    elif group_or_series == 'g':
        match data_choice: 
            case 'a': data_dict = [d.config.to_dict() for d in data]
            case 'g': data_dict = [d.parameters for d in data]
            case 'n': data_dict = [d.to_dict() for d in data]

        print("Data dict ", data_dict)
        # Find keys which exist in all data dicts
        data_common_keys = set(data_dict[0].keys())
        for d in data_dict[1:]:
            data_common_keys = data_common_keys.intersection(set(d.keys()))
        if len(data_common_keys) == 0: print('No common keys found'); return None, None
        print("Common keys ", data_common_keys)

        key_chosen = False
        group_by = input('Enter the config dict entry to group by: ')
        while not key_chosen:
            if group_by in data_common_keys:
                key_chosen = True
            else:
                group_by = input('Invalid key, try again: ')
        
        # Find the index of entries in all results, where the value of the key is the same
        series = []
        
            
    else:
        raise ValueError('Invalid group or series choice')

    print("Series ", series)

    return series, data
    
        


def heatmap_plot_results(results: ExperimentResults, plot: Plot):
    print("Plotting heatmap")

    num_noises  = len(results.experiment.noise_levels)
    num_results = len(list(results.all_results()))
    num_graphs  = len(results.experiment.graphs)
    num_algs    = len(results.experiment.algorithms)
    print("num_noises ", num_noises, " num_results ", num_results, " num_graphs ", num_graphs, " num_algs ", num_algs)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    match plot.params['x_axis']:
        case 'g':
            x_labels = [ExperimentResults.graph_descriptions(results.experiment.graphs)[str(g)] for g in results.experiment.graphs]
            accuracy_matrix = [[r[3].accuracy for r in results.all_results() if r[0] == g] for g in results.experiment.graphs]
        case 'a':
            x_labels = [ExperimentResults.algorithms_descriptions(results.experiment.algorithms)[a] for a in results.experiment.algorithms]
            accuracy_matrix = [[r[3].accuracy for r in results.all_results() if r[2] == a] for a in results.experiment.algorithms]
        case 'n':
            x_labels = [str(n.source_noise*100) + "% Noise" for n in results.experiment.noise_levels]
            accuracy_matrix = [[r[3].accuracy for r in results.all_results() if r[1] == n] for n in results.experiment.noise_levels]
        case 'c':
            x_labels, accuracy_matrix = prompt_custom_heatmap_axis(results)
        case _:
            raise ValueError('Invalid x-axis value')
        
    match plot.params['y_axis']:
        case 'g':
            y_labels = [ExperimentResults.graph_descriptions(results.experiment.graphs)[str(g)] for g in results.experiment.graphs]
            accuracy_matrix = [[r[3].accuracy for r in results.all_results() if r[0] == g] for g in results.experiment.graphs]
        case 'a':
            y_labels = [ExperimentResults.algorithms_descriptions(results.experiment.algorithms)[a] for a in results.experiment.algorithms]
            accuracy_matrix = [[r[3].accuracy for r in results.all_results() if r[2] == a] for a in results.experiment.algorithms]
        case 'n':
            y_labels = [str(n.source_noise*100) + "% Noise" for n in results.experiment.noise_levels]
            accuracy_matrix = [[r[3].accuracy for r in results.all_results() if r[1] == n] for n in results.experiment.noise_levels]
        case _:
            raise ValueError('Invalid y-axis value')
        
    print("x_labels ", x_labels, " y_labels ", y_labels, " accuracy_matrix ", accuracy_matrix)


    im = ax.imshow(accuracy_matrix)
    ax.set_xticks(np.arange(len(x_labels)), x_labels)
    ax.set_yticks(np.arange(len(y_labels)), y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
    [[ax.text(j, i, f'{accuracy_matrix[i][j]:10.2f}', ha="center", va="center", color="w") 
      for i in range(len(y_labels))] for j in range(len(x_labels))]

    ax.set_title("Accuracy heatmap")
    fig.tight_layout()
    plt.savefig(file.name + '_heatmap.png', dpi=300, bbox_inches = "tight")


if __name__ == '__main__':
    setup_readline()
    results, file = prompt_user_and_load_results()
    results, plot = prompt_user_for_analysis_options(results)
    file.close()

    if plot.plot_type == PlotType.BAR:
        bar_plot_results(results, plot)
    elif plot.plot_type == PlotType.HEATMAP:
        heatmap_plot_results(results, plot)
    else:
        raise ValueError('Invalid plot type')