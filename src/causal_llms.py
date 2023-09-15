import sys
import argparse
import scraping
import gpt
import pandas as pd
import re
from datetime import datetime
import benchmarks
import json
import os
import time

def sanitize_string(string, max_length=100):
    string = re.sub(r'[\\/:*?"<>|]', '_', string)
    return string[:max_length] if max_length else string


def causal_analysis(data, file_name=None, use_short_abstracts=False, max_abstract_length=200):

    print('CAUSAL ANALYSIS PROCESS')

    print(f'Starting at : {datetime.now().strftime("%H:%M:%S %d/%m/%Y")}')

    if file_name:
        file_name = sanitize_string(file_name)
    else:
        file_name = f'causal_analysis_results.csv'
    
    directory = f'../results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    file_name = f'{directory}/{file_name}'
    graphs_directory = f'{directory}/graphs'
    os.makedirs(directory, exist_ok=True)
    os.makedirs(graphs_directory, exist_ok=True)

    results = pd.DataFrame(columns=['id', 'title', 'abstract', 'exec_time'])

    print(len(data))

    for row in data.iterrows():
        row = row[1]
        if use_short_abstracts and len(row['abstract'].split(' ')) > max_abstract_length:
            continue

        title = sanitize_string(row['title'], 35)
        article_ref = f'{row["id"]}-{title}'

        start = time.time()
        print(f'\n-------- {row["title"]} --------\n')
        nodes, edges, cycles = gpt.causal_discovery_pipeline(article_ref, row['abstract'], use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_static_graph=False, graph_directory_name=graphs_directory, verbose=False)
        elapsed_seconds = time.time() - start

        new_row = pd.DataFrame({'id': row['id'], 'title': row['title'], 'abstract': row['abstract'], 'exec_time': time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))}, index=[0])
        results = pd.concat([results, new_row]).reset_index(drop=True)
        results.to_csv(file_name, index=False)


        graph_data = {'nodes': nodes, 'edges': edges, 'cycles': cycles}
        with open(f'{graphs_directory}/{article_ref}.json', "w") as json_file:
            json.dump(graph_data, json_file, indent=4)


    return results


def pubmed_scraping():
    print('SCRAPING PROCESS')
    print('------------------\n')

    scraping.main()


def scraping_and_causal_analysis():
    data = scraping.main(return_data=True)
    if data is None:
        print('ERROR: No data')
        sys.exit()

    causal_analysis(data)



def run_benchmarks(model=benchmarks.Algorithm.GPT):
    print('BENCHMARKS')
    print('------------------\n')

    benchmarks.run_benchmarks(model)


def evaluate_results(ground_truth, prediction, results_directory=None):
    print('EVALUATE RESULTS')
    # abbiamo un semplice script/funzioncina che prende in input due .json con grafi sugli stessi nodi 
    #   e misura le varie metriche che tu già consideri.
    with open(ground_truth, 'r') as json_file:
    # Parse the JSON data into a Python dictionary.
        gt_graph = json.load(json_file)
    with open(prediction, 'r') as json_file:
    # Parse the JSON data into a Python dictionary.
        pred_graph = json.load(json_file)

    directory = f'../evaluations/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}' if results_directory is None else results_directory
    os.makedirs(directory, exist_ok=True)

    shd, aupr, curve, precision, recall, f1, prediction_edges, missing_edges, extra_edges, correct_direction, incorrect_direction = benchmarks.evaluate_predictions(gt_graph['nodes'], gt_graph['edges'], pred_graph['edges'])

    results = pd.DataFrame({
            'SHD': shd,
            'Ground Truth edges': len(gt_graph['edges']),
            'Prediction edges': len(prediction_edges),
            'Missing edges': len(missing_edges),
            'Extra edges': len(extra_edges),
            'Correct direction': len(correct_direction),
            'Incorrect direction': len(incorrect_direction),
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUPR': aupr,
            'PRC point': [curve],
            'Prediction': [prediction_edges]
        }, index=[0])
    results.to_csv(f'{directory}/results.csv')


def run_example_test():
    print('EXAMPLE TEST')
    print('------------------\n')
    directory = f'../results/TEST - {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(directory , exist_ok=True)

    gpt.example_test(directory)
    print('\n--\nTEST COMPLETE')


class MyArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        if file is None:
            file = sys.stdout

        custom_help = """
Usage: causal_llms.py <action> [options]

Description:
  This script performs various tasks related to causal discovery.

Actions:
  ex       Run the example test.
  s        Run the scraping process.
  c        Perform causal analysis.
  sc       Run scraping and causal analysis.
  b        Run the benchmark tests.

Options:
  --help   Show this help message and exit.

Examples:
  python causal_llms.py ex                               # Run the example test.
  python causal_llms.py s                                # Run the scraping process.
  python causal_llms.py c --data-path </path/to/data>    # Perform causal analysis with specified data path.
  python causal_llms.py sc                               # Run scraping and causal analysis.
  python causal_llms.py b --algorithm {b|gpt}            # Run the benchmark tests with the specified algorithm.

The `algorithm` parameter specifies the algorithm to use for the benchmark tests.
The possible values are:
* `b`: Baseline algorithm
* `gpt`: GPT LLM
"""
        file.write(custom_help +"\n")



def main():

    parser = MyArgumentParser()
    parser.add_argument("action", choices=["ex", "s", "c", "sc","b", "e"], help="Action to perform.")
    parser.add_argument("--data-path", help="Path to the data for causal analysis.")
    parser.add_argument("--algorithm", help="Path to the algorithm for causal analysis on benchmarks.")


    try:

        args = parser.parse_args()

        # Check and use the parsed action
        if args.action == "b":
            if args.algorithm.upper() in [attr for attr in dir(benchmarks.Algorithm) if attr.isupper()]:
                run_benchmarks(model=getattr(benchmarks.Algorithm, args.algorithm.upper()))
            else:
                run_benchmarks()
        elif args.action == "ex":
            run_example_test()
        elif args.action == "s":
            pubmed_scraping()
        elif args.action == "c":
            if args.data_path:
                data = None
                try:
                    data = pd.read_csv(args.data_path)
                except FileNotFoundError:
                    print(f"CSV file not found: {args.data_path}")
                    return
                except pd.errors.ParserError:
                    print(f"Error parsing CSV file: {args.data_path}")
                    return
                except pd.errors.EmptyDataError:
                    print(f"CSV file is empty: {args.data_path}")
                    return
                except UnicodeDecodeError:
                    print(f"Error decoding CSV file: {args.data_path}")
                    return
                except PermissionError:
                    print(f"Permission error: {args.data_path}")
                    return
                except IOError:
                    print(f"I/O error: {args.data_path}")
                    return

                causal_analysis(data)
            else:
                print("Please provide the path to the data for causal analysis using the --data-path option.")
        elif args.action == "sc":
            scraping_and_causal_analysis()
        elif args.action == "e":
            gt = '../results/TEST - 2023-09-15_17-51-50/Example test.json'
            pred = '../results/TEST - 2023-09-15_17-51-50/Example test.json'
            gt = '../results/TEST - 2023-09-15_18-00-31/Example test.json'
            pred = '../results/TEST - 2023-09-15_18-00-31/Example test.json'
            evaluate_results(gt, pred)
        else:
            raise argparse.ArgumentError
    except argparse.ArgumentError:
        print("Invalid action. Use --help for available options.")

if __name__ == "__main__":
    main()