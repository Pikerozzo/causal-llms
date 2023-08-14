#!/usr/bin/env python
# coding: utf-8

import datetime
import subprocess
import pkgutil

packages_to_install = [
    'openai', 'pyvis', 'plotly', 'cdt', 'python-dotenv',
    'pandas', 'matplotlib', 'requests', 'bs4', 'lxml', 'tqdm', 'torch'
]

for package in packages_to_install:
    if not pkgutil.find_loader(package):
        subprocess.run(['pip', 'install', package, '--quiet'])

import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from itertools import permutations, combinations
import json

import networkx as nx
from pyvis.network import Network
import graph_tool.all as gt

from cdt.metrics import SHD, precision_recall

import os
from dotenv import load_dotenv, find_dotenv
import random


_ = load_dotenv(find_dotenv())

api_key  = os.getenv('IDSIA_OPENAI_API_KEY')
organization = os.getenv('IDSIA_ORGANIZATION')

openai.api_key = api_key
openai.organization = organization

models = openai.Model.list()
model_ids = [model['id'] for model in models['data']]

gpt_4 = 'gpt-4'
default_model = 'gpt-3.5-turbo'
use_gpt_4 = True
if use_gpt_4 and gpt_4 in model_ids:
    default_model = gpt_4

default_model


def gpt_request(system_msg, user_msg, model=default_model, temperature=0):
    if not system_msg or not user_msg:
        return None
    try:
        response = openai.ChatCompletion.create(model=model,
                                            messages=[
                                                {"role": "system", "content": system_msg},
                                                {"role": "user", "content": user_msg}], 
                                            temperature=temperature)
        return response.choices[0].message.content
    except:
        return None


def gpt_ner(text):
    system_msg = 'You are a helpful assistant for Named Entity Recognition of medical texts.' 
        
    user_msg = f'''
    You will be provided with an abstract of a medical research paper delimited by the <Text></Text> xml tags. 
    Please read the provided abstract carefully to comprehend the context and content. Analyze the provided text 
    and identify all the meaningful entities that could contribute to understanding cause-and-effect relationships 
    between factors such as diseases, medications, treatments, interventions, symptoms, outcomes, effects, or risk factors. 

    Avoid including entities that are synonyms or can be used interchangeably to already identified ones. For example, if text contains 
    both "hospital" and "medical center" (which are synonyms and can be used interchangeably) and you already extracted "hospital" as final entity, 
    do not include "medical center" as well.
    
    Your response should highlight entities that are crucial for establishing causal relationships in the medical context.

    Answer listing only the found entities within the tags <Answer><Entity>[entity1]</Entity><Entity>[entity2]</Entity></Answer> 
    (e.g., <Answer><Entity>diabetes</Entity><Entity>hypertension</Entity></Answer>).)
    
    Text:
    <Text>{text}</Text>
    '''
    
    response = gpt_request(system_msg, user_msg)
    if not response:
        return []
    
    answer_text = response
    
    soup = BeautifulSoup(answer_text, 'html.parser')
    entities = [entity.text for entity in soup.find_all('entity')]
    
    return entities


def pick_random_causal_verb():
    verbs = ['provokes', 'triggers', 'causes', 'leads to', 'induces', 'results in', 'generates', 'produces', 'stimulates', 'instigates', 'engenders', 'promotes', 'gives rise to', 'sparks']
    # return 'causes'
    return random.choice(verbs)


def gpt_causal_discovery(entities, text=None, use_pretrained_knowledge=False, reverse_variable_check=False, verbose=False):

    graph_edges = []    

    system_msg = 'You are a helpful assistant for causal reasoning and cause-and-effect relationship discovery.'
 
    intro_msg = f'''
You will be provided with {"an abstract of a medical research paper delimited by the <Text></Text> xml tags, and " if text else ""}\
a pair of entities delimited by the <Entity></Entity> xml tags representing medical entities {"extracted from the given abstract" if text else ""}, such
as medications, treatments, symptoms, diseases, outcomes, side effects, or other medical factors.
            {f"""
            Text:  
            <Text>{text}</Text>""" if text else ""}'''
    instructions_msg = f'''
{"Please read the provided abstract carefully to comprehend the context and content." if text else ""}
Examine the roles, interactions, and details surrounding the entities {"within the abstract" if text else ""}.
Based {"only " if text and not use_pretrained_knowledge else ""}on {"the information in the text " if text else ""}{"and " if text and use_pretrained_knowledge else ""}\
{"your pretrained knowledge" if use_pretrained_knowledge or not text else ""}, determine the most likely cause-and-effect \
relationship between the entities from the following listed options (A, B, C, D):\
    '''
    option_choice_msg = f'''
Your response should accurately reflect the likely causal connection between the two entities based on the 
information {"presented in the text" if text else ""} {"and any additional knowledge" if text and use_pretrained_knowledge else ""} {"you are aware of" if use_pretrained_knowledge or not text else ""}.
If no clear causal relationship is apparent, select the appropriate option accordingly.
Then provide your final answer within the tags <Answer>[answer]</Answer>, (e.g. <Answer>C</Answer>).
    '''
    option_choice_msg = f'''
Your response should analyze the situation in a step-by-step manner, ensuring the correctness of the ultimate conclusion, which should 
accurately reflect the likely causal connection between the two entities based on the 
information {"presented in the text" if text else ""} {"and any additional knowledge" if text and use_pretrained_knowledge else ""} {"you are aware of" if use_pretrained_knowledge or not text else ""}.
If no clear causal relationship is apparent, select the appropriate option accordingly.

Then provide your final answer within the tags <Answer>[answer]</Answer>, (e.g. <Answer>C</Answer>).
'''
    total_iterations = len(list(permutations(entities, 2))) if reverse_variable_check else len(list(combinations(entities, 2)))
    progress_bar = tqdm(total=total_iterations, desc="Progress")

    for i1, e1 in enumerate(entities):
        for i2, e2 in enumerate(entities):
            if i1 == i2 or (not reverse_variable_check and i1 >= i2):
                continue

            if verbose:
                print(f'{i1} = {e1}, {i2} = {e2}') # TODO remove

            options_with_random_verb = f'''\
            Options:
            A: "{e1}" {pick_random_causal_verb()} "{e2}"; 
            B: "{e2}" {pick_random_causal_verb()} "{e1}"; 
            C: "{e1}" and "{e2}" are not directly causally related; 
            D: there is a common factor that is the cause for both "{e1}" and "{e2}";\
            '''

            user_msg = f'''\
            {intro_msg}

            Entities:
            <Entity>{e1}</Entity>
            <Entity>{e2}</Entity>
            \
            {instructions_msg}
            {options_with_random_verb}
            \
            {option_choice_msg}
            '''

            response = gpt_request(system_msg, user_msg)
            if response:
                graph_edges.append(((e1, e2), response))
            
                if verbose:
                    print(graph_edges[-1]) # TODO remove
                    print('----------------------------------')
            
            progress_bar.update(1)

    progress_bar.close()
    
    return graph_edges


forward_arrow = '->'
forward_arrow_answer = 'A'
backward_arrow = '<-'
backward_arrow_answer = 'B'
no_arrow = ' '
no_arrow_answer = 'C'
bidirectional_arrow = '<->'
bidirectional_arrow_answer = 'D'

arrows = {forward_arrow_answer:forward_arrow, backward_arrow_answer:backward_arrow, no_arrow_answer:no_arrow, bidirectional_arrow_answer:bidirectional_arrow}

answer_pattern = re.compile(r'^([A-Z])\.')

def get_edge_answer(text):
    soup = BeautifulSoup(text, 'html.parser')
    answer = soup.find('answer').text

    if answer in arrows:
        return arrows[answer]

    match = answer_pattern.match(answer)
    if match:
        answer = match.group(1)

    if answer in arrows:
        return arrows[answer]
    
    return None


def print_edges(graph_edges):
    for (e1, e2), answer in graph_edges:
        try:
            print(f'{e1} {get_edge_answer(answer)} {e2}')
        except:
            print(f'{e1} ? {e2}')


def optimize_entities(entities, text=None):
    system_msg = 'You are a helpful assistant for medical entity optimization, by accurately identifying synonyms, redundant entities, or entities that can be used interchangeably'

    entities_text = '\n'.join([f'<Entity>{entity}</Entity>' for entity in entities])

    user_msg = f'''
You will be provided with {'an abstract of a medical research paper delimited by the <Text></Text> xml tags, and ' if text else ''} \
a list of named entities representing medical entities, each one of them delimited by the <Entity></Entity> xml tags. \

{f"""Text:
<Text>{text}</Text>""" if text else ""}

Entities:
{entities_text}

Your task is to optimize this entity list by identifying synonyms within the entities and grouping them accordingly. 
Your goal is to create a JSON object where the keys represent the root word entities, and each key is associated with 
an array of its synonyms, i.e., words or entities that can be used interchangeably to the root word.
If a root word entity has no synonyms, its value in the JSON should be an empty array.

Ensure that each entity appears only once in the dictionary, either as key (i.e. root word) or as element in the value arrays (i.e. the synonyms):
an entity must must not appear as key if it is the synonym (i.e. in the value array) of another entity, and the other way around (i.e. must not be 
in the value array of an entity if it is already a key of the JSON object).
An entity must not be a synonym of itself.

You should efficiently process the given list of entities and produce the desired dictionary structure.
The output JSON object should accurately represent the optimized entities and their synonyms based on the provided list.

Then provide your final JSON object answer within the tags <Answer>[json_object]</Answer>, (e.g. <Answer>{{
    "smoking": ["tobacco", "nicotine", "cigarettes", "cigar"],
    "cancer": ["lung cancer"],
    "tumors": []
}}</Answer>).

Follow the example below to understand the expected output.

Example:

Given the initial list of entities:
<Entity>smoking</Entity>
<Entity>lung cancer</Entity>
<Entity>tumors</Entity>
<Entity>cancer</Entity>
<Entity>tobacco</Entity>
<Entity>nicotine</Entity>
<Entity>cigarettes</Entity>
<Entity>cigar</Entity>

You should pair the synonyms, generate the following JSON object and provide it as your answer:
<Answer>
{{
    "smoking": ["tobacco", "nicotine", "cigarettes", "cigar"],
    "cancer": ["lung cancer"],
    "tumors": []
}}
</Answer>

Note that every entity appears only once in the output JSON object, either as key or as element in the value arrays.

After you have finished building the JSON object, check and make sure that every entity appears only once in the output JSON object, either as key or as element in the value arrays.
'''
    
    response = gpt_request(system_msg, user_msg)
    if response:
        soup = BeautifulSoup(response, 'html.parser')
        answer = soup.find('answer').text
        try:
            opt_entities = json.loads(answer)
            if opt_entities:
                return opt_entities
        except (json.JSONDecodeError, TypeError):
            pass

    return entities


answer_pattern = re.compile(r'^([A-Z])[.:]')

def extract_edge_answers(edges):
    edges_with_answers = []

    for (e1, e2), text in edges:
        try:
            soup = BeautifulSoup(text, 'html.parser')
            answer = soup.find('answer').text
            
            if answer in arrows:
                edges_with_answers.append(((e1, e2), answer))
                continue

            match = answer_pattern.match(answer)
            if match:
                if match.group(1) in arrows:
                    edges_with_answers.append(((e1, e2), match.group(1)))
                    continue

        except:
            continue

    return edges_with_answers


def check_edge_compatibility(answer1, answer2):
    return (arrows[answer1], arrows[answer2]) in [(forward_arrow, backward_arrow), (backward_arrow, forward_arrow), (no_arrow, no_arrow), (bidirectional_arrow, bidirectional_arrow)]

def check_invalid_answers(directed_edges):
    invalid_edges = []
    valid_edges = []
    temp_edges = []
    answers = {}
    for (n1, n2), answer in directed_edges:

        if (n1, n2) not in temp_edges and (n2, n1) not in temp_edges:
            temp_edges.append((n1, n2))
            answers[(n1, n2)] = answer
        elif (n1, n2) in temp_edges:
            if answers[(n1, n2)] != answer:
                invalid_edges.append((((n1, n2), answer), ((n2, n1), answers[(n2, n1)])))
            else:
                valid_edges.append(((n1, n2), answer))
            
            temp_edges.remove((n1, n2))
        elif (n2, n1) in temp_edges:
            if check_edge_compatibility(answers[(n2, n1)], answer):
                valid_edges.append(((n1, n2), answer))
            else:
                invalid_edges.append((((n1, n2), answer), ((n2, n1), answers[(n2, n1)])))
            
            temp_edges.remove((n2, n1))

    for n1, n2 in temp_edges:
        if (n1, n2) not in invalid_edges:
            invalid_edges.append((((n1, n2), answer), ((n2, n1), answers[(n2, n1)])))
    
    return valid_edges, invalid_edges


def get_textual_answers(e1, e2, ans):
    if ans == forward_arrow_answer:
        return f'"{e1}" causes "{e2}"'
    elif ans == backward_arrow_answer:
        return f'"{e2}" causes "{e1}"'
    elif ans == no_arrow_answer:
        return f'"{e1}" and "{e2}" are not causally related'
    elif ans == bidirectional_arrow_answer:
        return f'there is a common factor that is the cause for both "{e1}" and "{e2}"'
    else:
        return None


def correct_invalid_edges(invalid_edges, text=None, use_pretrained_knowledge=False):
    graph_edges = []

    if not invalid_edges:
        return []
    
    system_msg = 'You are a helpful assistant for causal reasoning and cause-and-effect relationship discovery.'
 
    intro_msg = f'''
You will be provided with {"an abstract of a medical research paper delimited by the <Text></Text> xml tags, and " if text else ""}\
a pair of entities delimited by the <Entity></Entity> xml tags representing medical entities {"extracted from the given abstract" if text else ""} (such
as medications, treatments, symptoms, diseases, outcomes, side effects, or other medical factors), and two answers you previously gave to this same request\
that are incoherent with each other, delimited by the <Answer></Answer> xml tags.
            {f"""
Text:  
<Text>{text}</Text>""" if text else ""}'''
    instructions_msg = f'''
{"Please read the provided abstract carefully to comprehend the context and content." if text else ""}
Consider the previous answers you gave to this same request that are incoherent with each other, and the entities they refer to in order to give a correct answer.
Examine the roles, interactions, and details surrounding the entities {"within the abstract" if text else ""}.
Based {"only " if text and not use_pretrained_knowledge else ""}on {"the information in the text " if text else ""}{"and " if text and use_pretrained_knowledge else ""}\
{"your pretrained knowledge" if use_pretrained_knowledge or not text else ""}, determine the most likely cause-and-effect \
relationship between the entities from the following listed options (A, B, C, D):\
    '''
    option_choice_msg = f'''
Your response should accurately reflect the likely causal connection between the two entities based on the 
information {"presented in the text" if text else ""} {"and any additional knowledge" if text and use_pretrained_knowledge else ""} {"you are aware of" if use_pretrained_knowledge or not text else ""}.
If no clear causal relationship is apparent, select the appropriate option accordingly.
Then provide your final answer within the tags <Answer>[answer]</Answer>, (e.g. <Answer>C</Answer>).
    '''
    option_choice_msg = f'''
Your response should analyze the situation in a step-by-step manner, ensuring the correctness of the ultimate conclusion, which should 
accurately reflect the likely causal connection between the two entities based on the 
information {"presented in the text" if text else ""} {"and any additional knowledge" if text and use_pretrained_knowledge else ""} {"you are aware of" if use_pretrained_knowledge or not text else ""}.
If no clear causal relationship is apparent, select the appropriate option accordingly.

Then provide your final answer within the tags <Answer>[answer]</Answer>, (e.g. <Answer>C</Answer>).
'''

    for ((e1, e2), answer1), ((e3, e4), answer2) in invalid_edges:       

        previous_answers_msg = f'''
        Previous incoherent answers:
        <Answer>{get_textual_answers(e1, e2, answer1)}</Answer>
        <Answer>{get_textual_answers(e3, e4, answer2)}</Answer>'''

        options_with_random_verb = f'''
        Options:
        A: "{e1}" {pick_random_causal_verb()} "{e2}"; 
        B: "{e2}" {pick_random_causal_verb()} "{e1}"; 
        C: "{e1}" and "{e2}" are not directly causally related; 
        D: there is a common factor that is the cause for both "{e1}" and "{e2}";'''

        user_msg = f'''\
        {intro_msg}

        Entities:
        <Entity>{e1}</Entity>
        <Entity>{e2}</Entity>

        {previous_answers_msg}
        \
        {instructions_msg}
        
        {options_with_random_verb}
        \
        {option_choice_msg}
        '''

        response = gpt_request(system_msg, user_msg)
        if response:
            graph_edges.append(((e1, e2), response))
            
    return graph_edges


def normalize_edge_direction(e1, e2, answer, graph):
    if answer in arrows:
        if arrows[answer] == forward_arrow:
            graph[e1].append(e2)
            return [(e1, e2)]
        elif arrows[answer] == backward_arrow:
            graph[e2].append(e1)   
            return [(e2, e1)]
        elif arrows[answer] == bidirectional_arrow:
            return [(e1, e2), (e2, e1)]
        else: 
            return None
    else: 
        return None


def preprocess_edges(edges):
    graph = {}
    processed_edges = []
    bidirected_edges = []

    for (n1, n2), answer in edges:

        if n1 not in graph:
            graph[n1] = []
        if n2 not in graph:
            graph[n2] = []
            
        direction = normalize_edge_direction(n1, n2, answer, graph)
        if direction:
            if len(direction) == 2:
                bidirected_edges.extend(direction)
            else:
                processed_edges.extend(direction)

    nodes = list(graph.keys())

    return nodes, processed_edges, bidirected_edges, graph


def find_cycles(nodes=[], edges=[]):
    if not nodes or not edges:
        return []

    g = gt.Graph(directed=True)

    nodes_ids = {}
    v_prop = g.new_vertex_property("string")
    for n in nodes:
        v = g.add_vertex()
        v_prop[v] = n
        nodes_ids[n] = v

    # e_prop = g.new_edge_property("string")
    for (n1, n2) in edges:
        e = g.add_edge(nodes_ids[n1], nodes_ids[n2])

    cycles = []
    for c in gt.all_circuits(g):
        cycles.append([v_prop[v] for v in c])

    return cycles


def build_graph(nodes, edges=[], bidirected_edges=[], cycles=[], plot_static_graph=True, directory_name='../graphs', graph_name='mygraph'):

    if plot_static_graph:
        plt.figure()
    G = nx.DiGraph()

    G.add_nodes_from(nodes)

    for e1, e2 in edges:
        G.add_edge(e1, e2, color='black', style='solid')

    cycles_edges = []
    for cycle in cycles:
        for i in range(len(cycle) - 1):
            G[cycle[i]][cycle[i + 1]]['color'] = 'red'
        G[cycle[-1]][cycle[0]]['color'] = 'red'

            # cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
            # cycle_edges.append((cycle[-1], cycle[0]))
            # cycles_edges.append(cycle_edges)

    
    for e1, e2 in bidirected_edges:
        G.add_edge(e1, e2, color='grey', style='dashed')

    if plot_static_graph:
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)

        edge_colors = [G.edges[edge]['color'] for edge in G.edges()]
        edge_styles = [G.edges[edge]['style'] for edge in G.edges()]

        nx.draw(G, pos, node_color='skyblue', node_size=1500,
                font_size=10, font_weight='bold', arrowsize=20, edge_color=edge_colors, style=edge_styles,
                width=2)
        plt.title(graph_name)
        plt.show()

    net = Network(directed=True, notebook=True)
    net.from_nx(G)
    net.force_atlas_2based()
    net.show_buttons(filter_=['physics'])
    os.makedirs(directory_name, exist_ok=True)
    net.save_graph(f'{directory_name}/{graph_name}.html')


def causal_discovery_pipeline(text_title, text, entities=[], use_text_in_causal_discovery=False, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_static_graph=True, graph_directory_name='../graphs', verbose=False):
    if verbose:
        print('Text:')
        print(text)
        print('--')

    if entities == []:
        entities = gpt_ner(text)
    else:
        if verbose:
            print('Skipping NER operation. Using provided entities.')
            print('--')


    print(f'Entities: ({len(entities)} = {entities})')

    if verbose:
        print(f'Entities: ({len(entities)})')
        print(entities)
        print('--')

    if optimize_found_entities:
        opt_entities = optimize_entities(entities, text=(text if use_text_in_entity_optimization else None))
        entities = list(opt_entities.keys())

        if verbose:
            print(f'Optimized Entities: ({len(entities)})')
            print(entities)

        print(f'Optimized Entities: ({len(entities)} = {entities})')
        

    graph_edges = gpt_causal_discovery(entities, text=(text if use_text_in_causal_discovery else None), use_pretrained_knowledge=use_LLM_pretrained_knowledge_in_causal_discovery, reverse_variable_check=reverse_edge_for_variable_check, verbose=verbose)

    print('GPT CAUSAL QUERY DONE')

    edges = extract_edge_answers(graph_edges)
    if verbose:
        print('Edges:')
        print(edges)
        print('--')

    if reverse_edge_for_variable_check:
        valid_edges, invalid_edges = check_invalid_answers(edges)
        if verbose:
            print('Valid Edges:')
            print(valid_edges)
            print('--')
            print('Invalid Edges:')
            print(invalid_edges)
            print('--')
        
        edge_correction_response = correct_invalid_edges(invalid_edges, text, use_pretrained_knowledge=use_LLM_pretrained_knowledge_in_causal_discovery)
        corrected_edges = extract_edge_answers(edge_correction_response)
        if verbose:
            print('Edge Correction Response:')
            print(corrected_edges)
            print('--')

        valid_edges.extend(corrected_edges)
        edges = valid_edges
        
        print('EDGE CORRECTION DONE')
    
    nodes, processed_edges, bidirected_edges, graph = preprocess_edges(edges)

    if verbose:
        print('Nodes:')
        print(nodes)
        print('--')
        print('Processed Edges:')
        print(processed_edges)
        print('--')

    cycles = []
    if search_cycles:
        start = time.time()
        print(f'Looking for cycles... {datetime.now().strftime("%H:%M:%S %d/%m/%Y")}')
        cycles = find_cycles(nodes=nodes, edges=processed_edges)
        elapsed_seconds = time.time() - start
        print(f'{len(cycles)} cycles found - exec time : {time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))}')
    build_graph(nodes=nodes, edges=processed_edges, bidirected_edges=bidirected_edges, cycles=cycles, plot_static_graph=plot_static_graph, directory_name=graph_directory_name, graph_name=text_title)
    
    if verbose:
        if cycles:
            print('GRAPH IS CYCLIC')
        else:
            print('Graph is acyclic')
        
        print('--')

    if verbose:
        print_edges(graph_edges)
    
    return nodes, processed_edges + bidirected_edges, cycles




# Sample text for test
def smoking_test():
    text = 'Smoking involves inhaling tobacco fumes and it causes lung cancer and tumors.'
    text_title = 'Smoking - test'
    return causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_graphs=False, plot_interactive_graph=False, verbose=True)










# Benchmarks
def plot_causal_graph(nodes, edges, title):

    if not nodes:
        return None
    
    # Create a graph
    G = nx.DiGraph()

    # Add nodes
    for node in nodes:
        G.add_node(node)

    # Add edges
    for e1, e2 in edges:
        G.add_edge(e1, e2)

    # Plot the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos)
    #add title to graph
    plt.title(title)


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def benchmark_evaluation(benchmark_title, ground_truth_nodes, ground_truth_edges, SHD_double_for_anticausal=True, plot_graphs=True, verbose=False):

    if ground_truth_nodes is None or ground_truth_edges is None:
        print("Ground truth nodes or edges are None.")
        return None, None, None
    
    nodes, prediction_edges, cycles = causal_discovery_pipeline(f'{benchmark_title} - Prediction', '', entities=ground_truth_nodes, use_text_in_causal_discovery=False, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=False, use_text_in_entity_optimization=False, search_cycles=True, plot_graphs=plot_graphs, plot_interactive_graph=False, verbose=verbose)
    print(prediction_edges)
    if plot_graphs:
        plot_causal_graph(ground_truth_nodes, ground_truth_edges, title=f'{benchmark_title} - Ground Truth')

    ground_truth_graph = nx.DiGraph()
    ground_truth_graph.add_nodes_from(nodes)
    ground_truth_graph.add_edges_from(ground_truth_edges)

    prediction_graph = nx.DiGraph()
    prediction_graph.add_nodes_from(nodes)
    prediction_graph.add_edges_from(prediction_edges)

    shd = SHD(ground_truth_graph, prediction_graph, double_for_anticausal=SHD_double_for_anticausal)

    if verbose:
        print(f'SHD = {shd}')

    aupr, curve = precision_recall(ground_truth_graph, prediction_graph)

    if plot_graphs:
        # Plot the precision-recall curve as a line plot
        precision_values = [point[0] for point in curve]
        recall_values = [point[1] for point in curve]
        plt.figure()
        plt.plot(recall_values, precision_values, marker='o', linestyle='-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Prediction Precision-Recall Curve')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.grid(True)

        # ideal line
        plt.plot([1.0, 1.0, 0.0], [0.0, 1.0, 1.0], linestyle='--')

        plt.show()


    if verbose:
        print(f"Area under the precision-recall curve: {aupr}")

    return shd, aupr, curve, prediction_edges, cycles


def precision_recall_curve_plot(titles, curves):
    fig = go.Figure()

    for i, curve_point in enumerate(curves):
        precision_values = [point[0] for point in curve_point]
        recall_values = [point[1] for point in curve_point]

        fig.add_trace(go.Scatter(
                x=recall_values,
                y=precision_values,
                text=f'F1 score = {f1_score(precision_values[1], recall_values[1]):.2f}',
                mode='lines+markers',
                name=titles[i]
            ))

    # ideal line
    fig.add_trace(go.Scatter(
                x=[0.0, 1.0, 1.0],
                y=[1.0, 1.0, 0.0],
                text='F1 score = 1.0',
                mode='lines+markers',
                name='Ideal PR line',
                line = dict(dash='dash'))
            )

    fig.update_layout(
        title='Prediction Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[-0.1, 1.1]),
        yaxis=dict(categoryorder='total ascending'),
    )

    fig.show()



def f1_score_hist(titles, curves):
    fig = go.Figure()

    best_points = [points[1] for points in curves]
    label = 'F1 score = '
    for i, (precision, recall) in enumerate(best_points):
        f1 = f1_score(precision, recall)

        fig.add_trace(go.Bar(
            y=[titles[i]],
            x=[f1],
            orientation='h',
            text=f'{f1:.2f}',
            textposition='inside',
            hoverinfo='x',
            name=titles[i]
        ))

    fig.update_layout(
        title='F1 Scores for Benchmarks',
        xaxis_title='F1 Score',
        yaxis_title='Benchmark',
        xaxis=dict(range=[0, 1.1]),
        yaxis=dict(categoryorder='total ascending'),
    )

    fig.show()



# Run causal discovery pipeline for all benchmarks
def run_benchmarks():
    ground_truth_graphs = [
                        ('Asia_benchmark', ['visit to Asia', 'tubercolosis', 'lung cancer', 'bronchitis', 'dyspnoea', 'smoking', 'positive X-ray'], [('visit to Asia', 'tubercolosis'), ('smoking', 'lung cancer'), ('smoking', 'bronchitis'), ('bronchitis', 'dyspnoea'), ('lung cancer', 'dyspnoea'), ('tubercolosis', 'dyspnoea'), ('lung cancer', 'positive X-ray'), ('tubercolosis', 'positive X-ray')]),
                        ('Smoking_benchmark', ['smoking', 'tobacco fumes', 'lung cancer', 'tumors'], [('smoking', 'tobacco fumes'), ('smoking', 'lung cancer'), ('smoking', 'tumors'), ('tobacco fumes', 'lung cancer'), ('tobacco fumes', 'tumors'), ('lung cancer', 'tumors'), ('tumors', 'lung cancer')]),
                        ('Alcohol_benchmark', ['alcohol', 'liver cirrhosis', 'death'], [('alcohol', 'liver cirrhosis'), ('liver cirrhosis', 'death'), ('alcohol', 'death')]),
                        ('Cancer_benchmark', ['smoking', 'respiratory disease', 'lung cancer', 'asbestos exposure'], [('smoking', 'respiratory disease'), ('respiratory disease', 'lung cancer'), ('asbestos exposure', 'lung cancer'), ('asbestos exposure', 'respiratory disease'), ('smoking', 'lung cancer')]),
                        ('Diabetes_benchmark', ['lack of exercise', 'body weight', 'diabetes', 'diet'], [('lack of exercise', 'body weight'), ('lack of exercise', 'diabetes'), ('body weight', 'diabetes'), ('diet', 'diabetes'), ('diet', 'body weight')]),
                        ('Obesity_benchmark', ['obesity', 'mortality', 'heart failure', 'heart defects'], [('obesity', 'mortality'), ('obesity', 'heart failure'), ('heart failure', 'mortality'), ('heart defects', 'heart failure'), ('heart defects', 'mortality')]),
                        ]

    titles = []
    sdhs = []
    auprs = []
    curves = []
    pred_edges = []
    pred_cycles = []

    for title, ground_truth_nodes, ground_truth_edges in ground_truth_graphs:
        shd, aupr, curve, prediction_edges, prediction_cycles = benchmark_evaluation(title, ground_truth_nodes, ground_truth_edges, plot_graphs=False, verbose=False)
        titles.append(title)
        sdhs.append(shd)
        auprs.append(aupr)
        curves.append(curve)
        pred_edges.append(prediction_edges)
        pred_cycles.append(prediction_cycles)
        print(f'{title} completed:')
        print(f'    SHD                  = {shd}')
        print(f'    Ground Truth edges   = {len(ground_truth_edges)} edges')
        print(f'    Prediction edges     = {len(prediction_edges)} edges')
        print(f'    Area PAC             = {aupr}')
        print(f'    PAC point            = {curve}')
        print(f'    Cycles               = {prediction_cycles}')
        print('')

    print('Benchmarks completed')

    precision_recall_curve_plot(titles, curves)
    f1_score_hist(titles, curves)