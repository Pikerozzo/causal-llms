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

    results = pd.DataFrame(columns=['id', 'title', 'abstract'])

    print(len(data))

    for row in data.iterrows():
        row = row[1]
        if use_short_abstracts and len(row['abstract'].split(' ')) > max_abstract_length:
            continue

        title = sanitize_string(row['title'], 35)
        article_ref = f'{row["id"]}-{title}'


        print(f'\n-------- {row["title"]} --------\n')
        nodes, edges, cycles = gpt.causal_discovery_pipeline(article_ref, row['abstract'], use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_static_graph=False, graph_directory_name=graphs_directory, verbose=False)
        new_row = pd.DataFrame({'id': row['id'], 'title': row['title'], 'abstract': row['abstract']}, index=[0])
        results = pd.concat([results, new_row]).reset_index(drop=True)
        results.to_csv(file_name, index=False)


        graph_data = {'nodes': [nodes], 'edges': [edges], 'cycles': [cycles]}
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



def run_example_test():
    print('EXAMPLE TEST')
    print('------------------\n')
    directory = f'../results/TEST - {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(directory , exist_ok=True)

    gpt.example_test(directory)



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
* `gpt`: GPT algorithm
"""
        file.write(custom_help +"\n")



def main():

    parser = MyArgumentParser()
    parser.add_argument("action", choices=["ex", "s", "c", "sc","b"], help="Action to perform.")
    parser.add_argument("--data-path", help="Path to the data for causal analysis.")
    parser.add_argument("--algorithm", help="Path to the data for causal analysis.")


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
        else:
            raise argparse.ArgumentError
    except argparse.ArgumentError:
        print("Invalid action. Use --help for available options.")

if __name__ == "__main__":
    main()