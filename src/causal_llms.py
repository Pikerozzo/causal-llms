import sys
import argparse
import scraping
import gpt
import pandas as pd
import re
import time
from datetime import datetime

import json

import os


def sanitize_string(string):
    return re.sub(r'[\\/:*?"<>|]', '_', string)


def causal_analysis(data, file_name=None, use_short_abstracts=False, max_abstract_length=200):
    
    print(f'Starting at : {datetime.now().strftime("%H:%M:%S %d/%m/%Y")}')

    if file_name:
        file_name = sanitize_string(file_name)
    else:
        file_name = f'causal_analysis_results.csv'
    
    directory = f'../results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    file_name = f'{directory}/{file_name}'
    graphs_directory = f'{directory}/graphs'
    os.makedirs(directory, exist_ok=True)

    results = pd.DataFrame(columns=['id', 'title', 'abstract'])

    print(len(data))

    for row in data.iterrows():
        row = row[1]
        if use_short_abstracts and len(row['abstract'].split(' ')) > max_abstract_length:
            continue

        title = sanitize_string(row['title'])
        max_title_length = 35
        tunc_title = title[:max_title_length] + (title[max_title_length:] and '..')
        article_ref = f'{row["id"]}-{tunc_title}'


        print(f'\n-------- {row["title"]} --------\n')
        # nodes, edges, cycles = gpt.causal_discovery_pipeline(article_ref, row['abstract'], use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_graphs=False, plot_interactive_graph=True, graph_directory_name=graphs_directory, verbose=False)
        nodes, edges, cycles = gpt.causal_discovery_pipeline(article_ref, row['abstract'], use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_static_graph=False, graph_directory_name=graphs_directory, verbose=False)
        new_row = pd.DataFrame({'id': row['id'], 'title': row['title'], 'abstract': row['abstract']}, index=[0])
        results = pd.concat([results, new_row]).reset_index(drop=True)
        results.to_csv(file_name, index=False)


        graph_data = {'nodes': [nodes], 'edges': [edges], 'cycles': [cycles]}
        with open(f'{graphs_directory}/{article_ref}.json', "w") as json_file:
            json.dump(graph_data, json_file, indent=4)


    return results




def causal_analysis_test():
    df = pd.read_csv('../data/dummy_data.csv')

    df = pd.read_csv('../data/pubmed_data.csv')
    df = df.sample(50)
    # df = df.loc[df['abstract'].apply(lambda x: len(x.split())).sort_values().head(5)]

    causal_analysis(df)




def scraping_and_causal_analysis():
    data = scraping.main(return_data=True)
    if data is None:
        print('ERROR: No data')
        sys.exit()

    causal_analysis(data)




    
#     # TODO - add command line parameters for operations
    


def scraping_and_causal_analysis_test():
    print('CAUSAL ANALYSIS TEST FROM SAVED DATA')

    data = pd.read_csv('../data/script_test.csv')
    if data is None:
        print('ERROR: No data')
        sys.exit()

    file_name = f'../data/graphs/causal_analysis_results{time.time().as_integer_ratio()[0]}.csv'
    results = pd.DataFrame(columns=['id', 'title', 'abstract', 'keywords', 'nodes', 'edges', 'cycles'])

    for row in data.iterrows():
        row = row[1]
        if len(row['abstract'].split(' ')) > 200:
            continue

        # id,title,abstract,keywords,pub_date,search_terms
        print(f'\n-------- {row["title"]} --------\n')
        nodes, edges, cycles = gpt.causal_discovery_pipeline(row['title'], row['abstract'], use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_graphs=False, plot_interactive_graph=True, verbose=False)
        new_row = pd.DataFrame({'id': row['id'], 'title': row['title'], 'abstract': row['abstract'], 'keywords': row['keywords'], 'nodes': [nodes], 'edges': [edges], 'cycles': [cycles]}, index=[0])
        results = pd.concat([results, new_row]).reset_index(drop=True)

        results.to_csv(file_name, index=False)

    results.to_csv(file_name, index=False)

def smoking_test():
    print('smoking test FROM SAVED DATA')

    data = pd.read_csv('../data/dummy_data.csv')
    if data is None:
        print('ERROR: No data')
        sys.exit()

    graph_results = pd.DataFrame(columns=['article_id', 'title', 'abstract', 'keywords', 'nodes', 'edges', 'cycles'])

    for row in data.iterrows():
        row = row[1]
        if len(row['abstract'].split(' ')) > 200:
            continue

        # print(row['title'])
        print(f'\n-------- {row["title"]} --------\n')
        nodes, edges, cycles = gpt.causal_discovery_pipeline(row['title'], row['abstract'], use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_graphs=False, plot_interactive_graph=True, verbose=False)
        graph_results = pd.concat([graph_results, pd.DataFrame({
            'article_id': row['article_id'], 'title': row['title'], 'abstract': row['abstract'], 'keywords': row['keywords'], 'nodes': [nodes], 'edges': [edges], 'cycles': [cycles]
            }, index=[0])]).reset_index(drop=True)
        
    
    graph_results.to_csv(f'../data/graphs/causal_analysis_results{time.time().as_integer_ratio()[0]}.csv', index=False)

def main():
    args = sys.argv[1:]  # Exclude the script name
    # print("Command line arguments:", args)

    # smoking_test()
    causal_analysis_test()
    # scraping_and_causal_analysis_test()
    # scraping_and_causal_analysis()

    # while True:
    #     print(f'''
    # Enter operation: 
    #         test(/t)
    #         scraping(/s)
    #         causal analysis(/c)
    #         scraping + causal analysis(/sc)''')
    #     term = input()

    #     if term == 't' or term == 'test':
    #         print('test')
    #         break
    #     elif term == 's' or term == 'scraping':
    #         print('scraping')
    #         break
    #     elif term == 'c' or term == 'causal analysis':
    #         print('causal analysis')
    #         break
    #     elif term == 'sc' or term == 'scraping + causal analysis':
    #         print('scraping + causal analysis')
    #         break
    #         break
    #     elif term == 'ciao':
    #         ciao()
    #         break
    #     else:
    #         print('ERROR: Invalid input')



    # parser = argparse.ArgumentParser(description="Call a specific method from the script.")
    # parser.add_argument("method", choices=["ciao"], help="Method to call")
    # parser.add_argument("--arg1", required=False, help="Argument 1")
    # parser.add_argument("--arg2", required=False, help="Argument 2")
    # args = parser.parse_args()

    # if args.method == "ciao":
    #     ciao()

def handle_help(): # TODO - add instructions for each command line parameter
    print("Usage: python your_script.py [options]")
    print("Options:")
    print("  --help: Show this help message")
    print("  --custom-option: Perform a custom action")


if __name__ == "__main__":
    
    main()
    
    # args = sys.argv[1:]  # Exclude the script name
    # # print("Command line arguments:", args)
    # if len(args) == 0:
    #     handle_help()
    #     sys.exit(0)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--help", action="store_true", help="Show help message")
    # parser.add_argument("--custom-option", action="store_true", help="Perform custom action")

    # args = parser.parse_args()

    # if args.help:
    #     handle_help()
    # elif args.custom_option:
    #     print("Custom action performed.")
    # else:
    #     main()