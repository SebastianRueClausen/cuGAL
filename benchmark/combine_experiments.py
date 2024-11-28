import os
import readline
import json
import numpy as np
from experiment import ExperimentResults, Experiment, Result

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
        json_file_path1 = input('Enter the path to the first JSON file: ')
        if os.path.isfile(json_file_path1):
            break
        else:
            print('File not found, try again.')

    while True:
        json_file_path2 = input('Enter the path to the second JSON file: ')
        if os.path.isfile(json_file_path2):
            break
        else:
            print('File not found, try again.')

    # Load JSON data
    file1 = open(json_file_path1, 'r')
    file2 = open(json_file_path2, 'r')
    results1 = ExperimentResults.from_dict(json.load(file1))
    results2 = ExperimentResults.from_dict(json.load(file2))
    return results1, results2

def combine_experiments(exp1: Experiment, exp2: Experiment) -> Experiment:
    # Combine the experiments
    algorithms = []
    for alg in exp1.algorithms + exp2.algorithms:
        if alg not in algorithms:
            algorithms.append(alg)
    graphs = []
    for graph in exp1.graphs + exp2.graphs:
        if graph not in graphs:
            graphs.append(graph)
    noise_levels = []
    for nl in exp1.noise_levels + exp2.noise_levels:
        if nl not in noise_levels:
            noise_levels.append(nl)
    debug = exp1.debug or exp2.debug
    seed = exp1.seed if exp1.seed is not None else exp2.seed
    save_alignment = exp1.save_alignment or exp2.save_alignment
    num_runs = min(exp1.num_runs, exp2.num_runs)

    return Experiment(
        algorithms=algorithms,
        graphs=graphs,
        noise_levels=noise_levels,
        debug=debug,
        seed=seed,
        save_alignment=save_alignment,
        num_runs=num_runs,
    )

def combine_results(results1: ExperimentResults, results2: ExperimentResults) -> ExperimentResults:
    combined_experiment = combine_experiments(results1.experiment, results2.experiment)

    all_results1 = results1.all_results()
    all_results2 = results2.all_results()
    # Create a dict from the tuple of (graph, algorithm, noise_level) to the result
    results_dict1 = {(str(result[0]), str(result[1]), str(result[2])): result[3] for result in all_results1}
    results_dict2 = {(str(result[0]), str(result[1]), str(result[2])): result[3] for result in all_results2}
    # Combine the dicts
    combined_results_dict = results_dict1.copy()
    combined_results_dict.update(results_dict2)

    print("results_dict1 ", len(results_dict1), results_dict1.keys(), "\n")
    print("results_dict2 ", len(results_dict2), results_dict2.keys(), "\n")

    print("both: \n", set(list(results_dict1.keys()) + list(results_dict2.keys())))

    print("combined_results_dict ", len(combined_results_dict), combined_results_dict, "\n")


    # Create the nested list from the dict. It will be a list of list of list of Result
    combined_results_list = []
    for graph in combined_experiment.graphs:
        noise_level_res = []
        for noise_level in combined_experiment.noise_levels:
            algorithm_res = []
            for algorithm in combined_experiment.algorithms:
                if (str(graph), str(noise_level), str(algorithm)) in combined_results_dict.keys():
                    algorithm_res.append(
                        combined_results_dict[(str(graph), str(noise_level), str(algorithm))]
                    )
                    print("Added a result\n")
                else:
                    algorithm_res.append([])
                    print("No result for ", str(graph), str(algorithm), str(noise_level), "\n")
            noise_level_res.append(algorithm_res)
        combined_results_list.append(noise_level_res)
            

    print("combined_results_list ", len(combined_results_list), combined_results_list, "\n")

    combined_results = ExperimentResults(
        experiment=combined_experiment,
        commit = results1.commit,
        time = results1.time,
        results=combined_results_list,
    )
    return combined_results

if '__main__' == __name__:
    setup_readline()
    results1, results2 = prompt_user_and_load_results()
    combined_results = combine_results(results1, results2)
    folder = "results_combined"
    if not os.path.exists(folder): os.makedirs(folder)
    combined_results.dump(folder)

