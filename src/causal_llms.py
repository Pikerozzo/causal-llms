import sys
import argparse
import scraping
import gpt_Copia
import pandas as pd
import re
import time

def scraping_and_causal_analysis():
    data = scraping.main(return_data=True)
    if data is None:
        print('ERROR: No data')
        sys.exit()
    
    # graph_results = pd.DataFrame(columns=['article_id', 'article_title', 'article_abstract', 'article_keywords', 'article_pub_date'])
    graph_results = pd.DataFrame(columns=['article_id', 'title', 'abstract', 'keywords', 'nodes', 'edges', 'cycles'])

    for row in data.iterrows():
        # id, title, abstract, keywords, pub_date, search_terms
        row = row[1]
        nodes, edges, cycles = gpt_Copia.causal_discovery_pipeline(row['title'], row['abstract'], use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_graphs=True, plot_interactive_graph=False, verbose=False)
        data = pd.concat([data, pd.DataFrame(
            {'id': row['id'], 'title': row['title'], 'abstract': row['abstract'], 'keywords': row['keywords'], 'nodes': nodes, 'edges': edges, 'cycles': cycles}, index=[0]
            )]).reset_index(drop=True)


    # TODO - add check for max length of abstract before proceeding with causal analysis
    #   words = re.findall(r'\b\w+\b', text)
    #   return len(words)
    
    # TODO - save to file causal analysis results
    
    # TODO - add command line parameters for operations
    
    # TODO - test on servers   





    data.to_csv(f'../data/graphs/{time.time().as_integer_ratio()[0]}.csv', index=False)


def ciao():
    print("CIAO FUNCTION")
    # sys.stdout.flush()

def main():
    args = sys.argv[1:]  # Exclude the script name
    print("Command line arguments:", args)

    while True:
        print(f'''Enter operation: 
            test(/t)
            scraping(/s)
            causal analysis(/c)
            scraping + causal analysis(/sc)''')
        term = input()

        if term == 't' or term == 'test':
            print('test')
            break
        elif term == 's' or term == 'scraping':
            print('scraping')
            break
        elif term == 'c' or term == 'causal analysis':
            print('causal analysis')
            break
        elif term == 'sc' or term == 'scraping + causal analysis':
            print('scraping + causal analysis')
            break
            break
        elif term == 'ciao':
            ciao()
            break
        else:
            print('ERROR: Invalid input')



    # parser = argparse.ArgumentParser(description="Call a specific method from the script.")
    # parser.add_argument("method", choices=["ciao"], help="Method to call")
    # parser.add_argument("--arg1", required=False, help="Argument 1")
    # parser.add_argument("--arg2", required=False, help="Argument 2")
    # args = parser.parse_args()

    # if args.method == "ciao":
    #     ciao()

def handle_help():
    print("Usage: python your_script.py [options]")
    print("Options:")
    print("  --help: Show this help message")
    print("  --custom-option: Perform a custom action")


if __name__ == "__main__":
    
    args = sys.argv[1:]  # Exclude the script name
    # print("Command line arguments:", args)
    if len(args) == 0:
        handle_help()
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--help", action="store_true", help="Show help message")
    parser.add_argument("--custom-option", action="store_true", help="Perform custom action")

    args = parser.parse_args()

    if args.help:
        handle_help()
    elif args.custom_option:
        print("Custom action performed.")
    else:
        main()