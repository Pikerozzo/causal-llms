#!/usr/bin/env python
# coding: utf-8


get_ipython().system('pip install openai')
get_ipython().system('pip install pyvis')
get_ipython().system('pip install plotly')
get_ipython().system('pip install python-dotenv')


import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
from bs4 import BeautifulSoup
import time
import json

import networkx as nx
from pyvis.network import Network

import os
from dotenv import load_dotenv, find_dotenv
import random


_ = load_dotenv(find_dotenv())

# api_key  = os.getenv('PIQUE_OPENAI_API_KEY')
# organization = os.getenv('PIQUE_SUPSI_ORGANIZATION')

api_key  = os.getenv('IDSIA_OPENAI_API_KEY')
organization = os.getenv('IDSIA_ORGANIZATION')

openai.api_key = api_key
openai.organization = organization


models = openai.Model.list()

ids = [model['id'] for model in models['data']]

gpt_4 = 'gpt-4'
default_model = 'gpt-3.5-turbo'
if gpt_4 in ids:
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


# In[19]:


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


# In[20]:


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


# In[21]:


from pyvis.network import Network

def build_graph(nodes, edges=[], bidirected_edges=[], search_and_highlight_cycles=True, plot_interactive_graph=True, plot_graph=True, graph_name='mygraph'):

    if plot_graph:
        plt.figure()
    G = nx.DiGraph()

    G.add_nodes_from(nodes)

    for e1, e2 in edges:
        G.add_edge(e1, e2, title=f'Text explanation for this edge ({e1}-{e2}) direction', color='black', style='solid')

    cycles = None
    cycles_edges = []
    if search_and_highlight_cycles:
        cycles = nx.recursive_simple_cycles(G)
        for cycle in cycles:
            for i in range(len(cycle) - 1):
                G[cycle[i]][cycle[i + 1]]['color'] = 'red'
            G[cycle[-1]][cycle[0]]['color'] = 'red'

            cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
            cycle_edges.append((cycle[-1], cycle[0]))
            cycles_edges.append(cycle_edges)

    for e1, e2 in bidirected_edges:
        G.add_edge(e1, e2, title=f'Text explanation for this edge ({e1}-{e2}) direction', color='grey', style='dashed')

    if plot_graph:
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

    if plot_interactive_graph:
        net = Network(directed=True, notebook=True)
        net.from_nx(G)
        net.force_atlas_2based()
        net.show_buttons(filter_=['physics'])
        net.show(f'../graphs/{graph_name}.html')

    if search_and_highlight_cycles:
        return cycles, cycles_edges
    else:
        return None, None


# In[22]:


def causal_discovery_pipeline(text_title, text, entities=[], use_text_in_causal_discovery=False, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_graphs=True, plot_interactive_graph=True, verbose=False):
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

    if verbose:
        print(f'Entities: ({len(entities)})')
        print(entities)
        print('--')

    if optimize_found_entities:
        if use_text_in_entity_optimization:
            opt_entities = optimize_entities(entities, text)
        else:
            opt_entities = optimize_entities(entities)

        entities = list(opt_entities.keys())



        if verbose:
            print(f'Optimized Entities: ({len(entities)})')
            print(entities)

    if use_text_in_causal_discovery:
        graph_edges = gpt_causal_discovery(entities, text, use_pretrained_knowledge=use_LLM_pretrained_knowledge_in_causal_discovery, reverse_variable_check=reverse_edge_for_variable_check, verbose=verbose)
    else:
        graph_edges = gpt_causal_discovery(entities, use_pretrained_knowledge=use_LLM_pretrained_knowledge_in_causal_discovery, reverse_variable_check=reverse_edge_for_variable_check, verbose=verbose)


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
    
    nodes, processed_edges, bidirected_edges, graph = preprocess_edges(edges)

    if verbose:
        print('Nodes:')
        print(nodes)
        print('--')
        print('Processed Edges:')
        print(processed_edges)
        print('--')

    cycles, cycles_edges = build_graph(nodes=nodes, edges=processed_edges, bidirected_edges=bidirected_edges, search_and_highlight_cycles=search_cycles, plot_interactive_graph=plot_interactive_graph, plot_graph=plot_graphs, graph_name=text_title)
    if verbose:
        if cycles:
            print('GRAPH IS CYCLIC')
        else:
            print('Graph is acyclic')
        
        print('--')

    if verbose:
        print_edges(graph_edges)
    
    return nodes, processed_edges + bidirected_edges, cycles


# In[23]:


def smoking_test():
    text = 'Smoking involves inhaling tobacco fumes and it causes lung cancer and tumors.'
    text_title = 'Smoking - test'
    causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_graphs=True, plot_interactive_graph=False, verbose=True)


# In[24]:


smoking_test()


# # Tests

# In[21]:


text = '\'X\' is the main cause for \'Y\'; \'Y\' causes \'Z\'; \'Z\' is the main cause for \'X\''
text = '\'X\' is caused by \'Z\'; \'X\' provokes \'Y\'; \'Y\' causes \'Z\''
text_title = 'cyclic_graph'
nodes, edges = causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=False, use_text_in_entity_optimization=False, verbose=True)


# In[161]:


import time
text = 'Smoking involves inhaling tobacco fumes; it can cause lung cancer and tumors.'
ground_truth_nodes = ['visit to Asia', 'tubercolosis', 'lung cancer', 'bronchitis', 'dyspnoea', 'smoking', 'positive X-ray']
text = 'Shortness-of-breath (dyspnoea) may be due to tuberculosis, lung cancer or bronchitis, or none of them, or more than one of them. A recent visit to Asia increases the chances of tuberculosis, while smoking is known to be a risk factor for both lung cancer and bronchitis. The results of a single chest X-ray do not discriminate between lung cancer and tuberculosis, as neither does the presence or absence of dyspnoea.'
text_title = f'Asia_{time.time().as_integer_ratio()[0]}'
nodes, edges, cycles = _causal_discovery_pipeline(text_title, text, entities=ground_truth_nodes, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=False, use_text_in_entity_optimization=False, search_cycles=True, plot_interactive_graph=True, verbose=True)


# In[33]:


import time
ground_truth_nodes = ['smoking', 'tobacco fumes', 'lung cancer', 'tumors']
text = 'Smoking involves inhaling tobacco fumes, and it can cause lung cancer and tumors.'
text_title = f'Benchmark_Smoking-{time.time().as_integer_ratio()[0]}'
nodes, edges, cycles = _causal_discovery_pipeline(text_title, text, entities=ground_truth_nodes, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=True, optimize_found_entities=False, use_text_in_entity_optimization=False, search_cycles=True, plot_interactive_graph=False, verbose=True)


# In[29]:


import time
ground_truth_nodes = ['smoking', 'tobacco fumes', 'lung cancer', 'tumors']
text = 'Smoking involves inhaling tobacco fumes; it can cause lung cancer and tumors.'
text_title = f'Benchmark_Smoking-{time.time().as_integer_ratio()[0]}'
nodes, edges, cycles = _causal_discovery_pipeline(text_title, text, entities=ground_truth_nodes, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=False, use_text_in_entity_optimization=False, search_cycles=True, plot_interactive_graph=True, verbose=True)


# # Misc (Kahn, old functions, ...)

# In[73]:


def ___has_cycle(graph):
  """
  Checks if a directed graph has a cycle.

  Args:
    graph: A directed graph represented as a dictionary. The keys of the dictionary are the nodes in the graph, and the values are lists of the nodes that the given node points to.

        graph = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": []
        }
    
  Returns:
    True if the graph has a cycle, False otherwise.
  """

  visited = set()
  stack = []

  for node in graph:

    print(f'node = {node}')  # delete


    # if node in visited:
    #   return True

    # stack.append(node)
    # # visited.add(node)
    # visited.add(node)

    
    if node not in visited:
      visited.add(node)
      stack.append(node)

    while stack:
      print(f'stack = {stack}')  # delete
      print(f'visited = {visited}')  # delete
      
      current_node = stack.pop()

      print(f'current_node = {current_node}')  # delete


      for neighbor in graph[current_node]:

        print(f'neighbor = {neighbor}')  # delete
        
        if neighbor in visited:
          return True

        stack.append(neighbor)
        visited.add(neighbor)

  return False


# {0: [1], 1: [2], 2: [3], 3: []}

# node = 0
# stack = [0]
# current_node = 0
# neighbor = 1
# stack = [1]
# current_node = 1
# neighbor = 2
# stack = [2]
# current_node = 2
# neighbor = 3
# stack = [3]
# current_node = 3
# node = 1




# {0: [1, 2], 1: [2], 2: [3, 5], 3: [4, 5]}

#  node = 0
#  current_node = 0
#  current_node = 2
#  current_node = 5


# In[72]:


def __has_cycle(graph):
  """
  Checks if a directed graph has a cycle.

  Args:
    graph: A directed graph represented as a dictionary. The keys of the dictionary are the nodes in the graph, and the values are lists of the nodes that the given node points to.

        graph = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": []
        }
    
  Returns:
    True if the graph has a cycle, False otherwise.
  """

  visited = set()
  stack = []

  for node in graph:

    print(f'node = {node}')  # delete


    # if node in visited:
    #   return True

    # stack.append(node)
    # # visited.add(node)
    # visited.add(node)

    
    if node not in visited:
      visited.add(node)
      stack.append(node)

    while stack:
      print(f'stack = {stack}')  # delete
      print(f'visited = {visited}')  # delete
      
      current_node = stack.pop()

      print(f'current_node = {current_node}')  # delete
      
      temp_visited = visited

      for neighbor in graph[current_node]:

        print(f'neighbor = {neighbor}')  # delete
        
        if neighbor in temp_visited:
          return True

        stack.append(neighbor)
        temp_visited.add(neighbor)
        # visited.add(neighbor)

  return False


# {0: [1], 1: [2], 2: [3], 3: []}

# node = 0
# stack = [0]
# current_node = 0
# neighbor = 1
# stack = [1]
# current_node = 1
# neighbor = 2
# stack = [2]
# current_node = 2
# neighbor = 3
# stack = [3]
# current_node = 3
# node = 1




# {0: [1, 2], 1: [2], 2: [3, 5], 3: [4, 5]}

#  node = 0
#  current_node = 0
#  current_node = 2
#  current_node = 5


# In[25]:


def __prepare_graph_for_acyclicity_check(edges):
    graph = {}
    nodes = set()
    nodes_without_incoming_edges = set()
    nodes_with_incoming_edges = set()

    for (n1, n2), answer in edges:
        nodes.add(n1)
        if n1 not in graph:
            graph[n1] = []
        if n2 not in graph:
            graph[n2] = []
            
        if answer == 'A':
            graph[n1].append(n2)
            nodes_with_incoming_edges.add(n2)
        if answer == 'B':
            graph[n2].append(n1) 
            nodes_with_incoming_edges.add(n1)

    nodes_without_incoming_edges = nodes - nodes_with_incoming_edges
    # print(f'{nodes} - {nodes_with_incoming_edges} = {nodes_without_incoming_edges}')
        

    return graph, nodes_without_incoming_edges


# In[64]:


def _prepare_graph_for_acyclicity_check(edges):
    graph = {}
    nodes = set()
    processed_edges = set()

    for (n1, n2), answer in edges:
        nodes.add(n1)
        nodes.add(n2)

        if n1 not in graph:
            graph[n1] = []
        if n2 not in graph:
            graph[n2] = []
            
        if answer == 'A':
            graph[n1].append(n2)
            processed_edges.add((n1, n2))
        if answer == 'B':
            graph[n2].append(n1)       
            processed_edges.add((n2, n1))
        elif answer == 'D':
            processed_edges.add((n1, n2))
            processed_edges.add((n2, n1))

    return nodes, processed_edges, graph


# In[71]:


def _has_cycle(graph):
    def dfs(node, visited, recursion_stack):
        if node in recursion_stack:
            return True
        if node in visited:
            return False

        visited.add(node)
        recursion_stack.add(node)

        for neighbor in graph.get(node, []):
            if dfs(neighbor, visited, recursion_stack):
                return True

        recursion_stack.remove(node)
        return False

    visited = set()
    for node in graph:
        if node not in visited:
            if dfs(node, visited, set()):
                return True

    return False


# In[72]:


edges = [(('X', 'Y'), 'A'), (('Y', 'Z'), 'A'), (('T', 'Z'), 'B'), (('X', 'T'), 'C')]        # no cycle
# edges = [(('X', 'Y'), 'A'), (('Y', 'Z'), 'A'), (('T', 'Z'), 'B'), (('X', 'T'), 'B'), (('T', 'A'), 'A')]        # has cycle

import random
import string

# generate random edges
edges = []
nodes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
nodes = ['1', '2', '3', '4', '5']

for i, n1 in enumerate(nodes):
    for j, n2 in enumerate(nodes):
        if n1 == n2 or i >= j:
            continue
        answer = random.choice(['A', 'B', 'C', 'D'])
        edges.append(((n1, n2), answer))

print(edges)

nodes, edges, graph = _prepare_graph_for_acyclicity_check(edges)
plot_interactive_graph(nodes, edges)
plot_graph(nodes, edges)
_has_cycle(graph)


# TODO : DFS instead of Kahn algorithm (?)
# TODO : update documentation


# In[6]:


def preprocess_edges(edges):
    nodes = set()
    processed_edges = set()
    directed_edges = set()
    nodes_with_incoming_edges = set()

    for (n1, n2), answer in edges:

        nodes.add(n1)
        nodes.add(n2)

        if answer == 'A':
            processed_edges.add((n1, n2))
            directed_edges.add((n1, n2))
            nodes_with_incoming_edges.add(n2)
        elif answer == 'B':
            processed_edges.add((n2, n1))
            directed_edges.add((n2, n1))
            nodes_with_incoming_edges.add(n1)
        elif answer == 'D':
            processed_edges.add((n1, n2))
            processed_edges.add((n2, n1))
            

    nodes_without_incoming_edges = nodes - nodes_with_incoming_edges        

    return nodes, processed_edges, directed_edges, nodes_without_incoming_edges





def prepare_graph_for_acyclicity_check(edges):
    nodes = set()
    processed_edges = set()
    nodes_with_incoming_edges = set()

    for (n1, n2), answer in edges:
        nodes.add(n1)
            
        if answer == 'A':
            nodes_with_incoming_edges.add(n2)
            processed_edges.add((n1, n2))
        if answer == 'B':
            nodes_with_incoming_edges.add(n1)
            processed_edges.add((n2, n1))

    nodes_without_incoming_edges = nodes - nodes_with_incoming_edges        

    return nodes, processed_edges, nodes_without_incoming_edges



# TODO 1: unify 'preprocess_edges' and 'prepare_graph_for_acyclicity_check', they are doing almost the same thing (the latter has the additional steps 
#       of 1. collecting nodes without incoming edges and 2. edges to be considered for the acyclicity check --> only single direction edges and no bidirected edges)

# TODO 2: decide what to do if graph is not acyclical (--> re-query to remove cycle ??)

# TODO 3: update documentation with the acyclicity check

# TODO 4: update documentation with introductive theory on causal analysis and discovery

# TODO 5: benchmarks
        # 5.1. in code
        # 5.2. in documentation


# In[5]:


# def kahn_acyclicity_check(graph, nodes_without_incoming_edges = set()):
def kahn_acyclicity_check(edges, nodes_without_incoming_edges = set()):

    '''
    Kahn's algorithm for checking acyclicity of a directed graph.

    Args:
        graph: A directed graph represented as a dictionary. The keys of the dictionary are the nodes in the graph, and the values are lists of the nodes that the given node points to.
        nodes_without_incoming_edges: A set of nodes that have no incoming edges.

    Returns:
        True if the graph is acyclic, False otherwise.
    '''

    '''
    L ← Empty list that will contain the sorted elements
    S ← Set of all nodes with no incoming edge

    while S is not empty do
        remove a node n from S
        add n to L
        for each node m with an edge e from n to m do
            remove edge e from the graph
            if m has no other incoming edges then
                insert m into S

    if graph has edges then
        return error   (graph has at least one cycle)
    else 
        return L   (a topologically sorted order)

    '''

    L = []

    while nodes_without_incoming_edges:
        n = nodes_without_incoming_edges.pop()
        L.append(n)

        # for m children of n
        for m in [tup for tup in edges if tup[0] == n]:

            edges.remove((n, m[1]))
            # if m has no more parent nodes
            if not [tup for tup in edges if tup[1] == m[1]]:
                nodes_without_incoming_edges.add(m[1])


    if edges:
        return True
    else:
        return False


# In[15]:


edges = [(('X', 'Y'), 'A'), (('Y', 'Z'), 'A'), (('T', 'Z'), 'B'), (('X', 'T'), 'C')]        # no cycle
edges = [(('X', 'Y'), 'A'), (('Y', 'Z'), 'A'), (('T', 'Z'), 'B'), (('X', 'T'), 'B'), (('T', 'A'), 'A')]        # has cycle


nodes, processed_edges, directed_edges, nodes_without_incoming_edges = preprocess_edges(edges)
print(kahn_acyclicity_check(directed_edges, nodes_without_incoming_edges))
plot_interactive_graph(nodes, processed_edges)


# In[115]:


edges = [(0,1), (1,2), (1,3), (2,3), (3,0), (3, 1)]

child_node_edges = [tup for tup in edges if tup[0] == 1]
parent_node_edges = [tup for tup in edges if tup[1] == 1]

# print(f'child_node_edges = {child_node_edges}')
# print(f'parent_node_edges = {parent_node_edges}')


# edges = [((0, 1), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((0, 3), 'C')]        # no cycle
# # edges = [((0, 1), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((0, 3), 'B')]        # has cycle

# graph, nodes_without_incoming_edges = prepare_graph_for_acyclicity_check(edges)

# graph = [(0,1), (1,2), (2,3)]               # no cycle
# # graph = [(0,1), (1,2), (2,3), (3,0)]        # has cycle


edges = [(('X', 'Y'), 'A'), (('Y', 'Z'), 'A'), (('T', 'Z'), 'B'), (('X', 'T'), 'C')]        # no cycle
edges = [(('X', 'Y'), 'A'), (('Y', 'Z'), 'A'), (('T', 'Z'), 'B'), (('X', 'T'), 'B')]        # has cycle

graph, nodes_without_incoming_edges = prepare_graph_for_acyclicity_check(edges)

graph = [('X','Y'), ('Y','Z'), ('Z','T')]               # no cycle
graph = [('X','Y'), ('Y','Z'), ('Z','T'), ('T','X')]        # has cycle

kahn_acyclicity_check(graph, nodes_without_incoming_edges)


# In[88]:


edges = [((1, 0), 'B'), ((0, 2), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((3, 4), 'A'), ((3, 5), 'A'), ((2, 5), 'B'), ((0, 3), 'C'), ((0, 4), 'C'), ((5, 0), 'C')]        # has cycle
edges = [((1, 0), 'B'), ((0, 2), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((3, 4), 'A'), ((3, 5), 'A'), ((2, 5), 'A'), ((0, 3), 'C'), ((0, 4), 'C'), ((5, 0), 'C')]        # no cycle

edges = [((0, 1), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((0, 3), 'C')]        # no cycle
# edges = [((0, 1), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((0, 3), 'B')]        # has cycle

print(prepare_graph_for_acyclicity_check(edges))


# In[76]:


# (('X', 'Y'), 'A'), ('X', 'Y', 'A')

graph = {
    0: [1, 2],
    1: [2],
    2: [3],
    3: [4, 5],
    4: [],
    5: [2]
}



def check_graph_acyclicity(edges):
    graph = {}

    for (n1, n2), answer in edges:
        if n1 not in graph:
            graph[n1] = []
        if n2 not in graph:
            graph[n2] = []
        if answer == 'A':
            # if n1 not in graph:
            #     graph[n1] = []
            graph[n1].append(n2)
        if answer == 'B':
            # if n2 not in graph:
            #     graph[n2] = []
            graph[n2].append(n1) 

    return graph


edges = [((1, 0), 'B'), ((0, 2), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((3, 4), 'A'), ((3, 5), 'A'), ((2, 5), 'B'), ((0, 3), 'C'), ((0, 4), 'C'), ((5, 0), 'C')]        # has cycle
# edges = [((1, 0), 'B'), ((0, 2), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((3, 4), 'A'), ((3, 5), 'A'), ((2, 5), 'A'), ((0, 3), 'C'), ((0, 4), 'C'), ((5, 0), 'C')]        # no cycle

# edges = [((0, 1), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((0, 3), 'C')]        # no cycle
# edges = [((0, 1), 'A'), ((1, 2), 'A'), ((3, 2), 'B'), ((0, 3), 'B')]        # has cycle

print(check_graph_acyclicity(edges))

has_cycle(check_graph_acyclicity(edges))


# In[40]:


edges = [('0', '1'), ('0', '2'), ('1', '2'), ('2', '3'), ('2', '5'), ('3', '4'), ('3', '5')]        # no cycle
plot_interactive_graph(['0', '1', '2', '3', '4', '5'], edges, name='test_no_cycle')

edges = [('0', '1'), ('0', '2'), ('1', '2'), ('2', '3'), ('3', '4'), ('3', '5'), ('5', '2')]        # has cycle
plot_interactive_graph(['0', '1', '2', '3', '4', '5'], edges, name='test_cycle')


# In[22]:


text = 'Shortness-of-breath (dyspnoea) may be due to tuberculosis, lung cancer or bronchitis, or none of them, or more than one of them. A recent visit to Asia increases the chances of tuberculosis, while smoking is known to be a risk factor for both lung cancer and bronchitis. The results of a single chest X-ray do not discriminate between lung cancer and tuberculosis, as neither does the presence or absence of dyspnoea.'

entities = gpt_ner(text)

print(f'Entities: ({len(entities)})')
print(entities)
print('--')

opt_entities = optimize_entities(entities, text)
# opt_entities = optimize_entities(entities)
  
print(f'Optimized Entities: ({len(opt_entities)})')
print(opt_entities)
print('--')


# In[16]:


# pre build_graph function (with grey, dashed bidirectional edges and cycle highlighting)

def has_cycle(graph):
    """
    Checks if a directed graph has a cycle.

    Args:
        graph: A directed graph represented as a dictionary. The keys of the dictionary are the nodes in the graph, and the values are lists of the nodes that the given node points to.

            graph = {
            "A": ["B", "C"],
            "B": ["C"],
            "C": []
            }
        
    Returns:
        True if the graph has a cycle, False otherwise.
  """

  # visits 

    def dfs(node, visited, recursion_stack):
        if node in recursion_stack:
            return True
        if node in visited:
            return False

        visited.add(node)
        recursion_stack.add(node)

        for neighbor in graph.get(node, []):
            if dfs(neighbor, visited, recursion_stack):
                return True

        recursion_stack.remove(node)
        return False

    visited = set()
    for node in graph:
        if node not in visited:
            if dfs(node, visited, set()):
                return True

    return False


# In[2]:


# pre build_graph function (with grey, dashed bidirectional edges and cycle highlighting)

import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(nodes, edges):

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


# In[138]:


# pre build_graph function (with grey, dashed bidirectional edges and cycle highlighting)

from pyvis.network import Network

def plot_interactive_graph(nodes, edges, name='mygraph'):
    
    if not nodes:
        return None
    
    net = Network(directed=True, notebook=True)
    net.force_atlas_2based()
    net.show_buttons(filter_=['physics']) 

    node_ids = {}

    for i, node in enumerate(nodes):
        net.add_node(i, label=node)
        node_ids[node] = i
    
    for e1, e2 in edges:
        net.add_edge(source=node_ids[e1], to=node_ids[e2], label='TEST LABEL')

    net.show(f'../graphs/{name}.html')


# In[136]:


# pre build_graph function (with grey, dashed bidirectional edges and cycle highlighting)

def causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=False, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, verbose=False):
    if verbose:
        print('Text:')
        print(text)
        print('--')

    entities = gpt_ner(text)

    if verbose:
        print(f'Entities: ({len(entities)})')
        print(entities)
        print('--')

    if optimize_found_entities:
        if use_text_in_entity_optimization:
            entities = optimize_entities(entities, text)
        else:
            entities = optimize_entities(entities)

        if verbose:
            print(f'Optimized Entities: ({len(entities)})')
            print(entities)
            print('--')

    if use_text_in_causal_discovery:
        graph_edges = gpt_causal_discovery(entities, text, use_pretrained_knowledge=use_LLM_pretrained_knowledge_in_causal_discovery, reverse_variable_check=reverse_edge_for_variable_check)
    else:
        graph_edges = gpt_causal_discovery(entities, use_pretrained_knowledge=use_LLM_pretrained_knowledge_in_causal_discovery, reverse_variable_check=reverse_edge_for_variable_check)


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
    
    nodes, processed_edges, graph = preprocess_edges(edges)
    is_graph_acyclic = not has_cycle(graph)

    if verbose:
        if is_graph_acyclic:
            print('Graph is acyclic')
        else:
            print('Graph is cyclic')
        
        print('--')

    if verbose:
        print('Nodes:')
        print(nodes)
        print('--')
        print('Processed Edges:')
        print(processed_edges)
        print('--')

    plot_graph(nodes, processed_edges)
    plot_interactive_graph(nodes, processed_edges, name=text_title)
    print_edges(graph_edges) # temporary edge print

    return nodes, processed_edges


# # Causal Discovery Tests

# In[21]:


# text = 'Smoking is a major cause of lung cancer. Tobacco smoke contains harmful substances that can lead to tumor formation in the lungs. Quitting smoking reduces the risk of lung cancer and improves overall health.'
text = 'Smoking involves inhaling tobacco fumes; it can cause lung cancer and tumors.'
text_title = 'Smoking'
causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=True, optimize_found_entities=False, use_text_in_entity_optimization=True, verbose=True)


# In[20]:


# text = 'Smoking is a major cause of lung cancer. Tobacco smoke contains harmful substances that can lead to tumor formation in the lungs. Quitting smoking reduces the risk of lung cancer and improves overall health.'
text = 'Smoking involves inhaling tobacco fumes; it can cause lung cancer and tumors.'
text_title = 'Smoking'
causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=True, optimize_found_entities=True, use_text_in_entity_optimization=True, verbose=True)


# # Benchmarks

# ## Evaluation Metrics
# 
# for benchmark graphs (without NER step? directly with correct nodes)

# Precision is defined by: _Pr = tp / (tp + fp)_ and directly denotes the total classification accuracy given a confidence threshold. \
# On the other hand, Recall is defined by: _Re = tp / (tp + fn)_ and denotes misclassification given a threshold

# ### Structural Hamming Distance

# In[175]:


get_ipython().system('pip install cdt')


# In[148]:


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


# In[149]:


ground_truth_nodes = ['visit to Asia', 'tubercolosis', 'lung cancer', 'bronchitis', 'dyspnoea', 'smoking', 'positive X-ray']
ground_truth_edges = [('visit to Asia', 'tubercolosis'), ('smoking', 'lung cancer'), ('smoking', 'bronchitis'), ('bronchitis', 'dyspnoea'), ('lung cancer', 'dyspnoea'), ('tubercolosis', 'dyspnoea'), ('lung cancer', 'positive X-ray'), ('tubercolosis', 'positive X-ray')]
# plot_interactive_graph(ground_truth_nodes, ground_truth_edges, name='Asia - Ground Truth')
plot_causal_graph(ground_truth_nodes, ground_truth_edges, title='Asia - Ground Truth')


# In[150]:


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


# In[151]:


from cdt.metrics import SHD, precision_recall
from numpy.random import randint

def benchmark_evaluation(benchmark_title, ground_truth_nodes, ground_truth_edges, SHD_double_for_anticausal=True, plot_graphs=True, verbose=False):

    if ground_truth_nodes is None or ground_truth_edges is None:
        print("Ground truth nodes or edges are None.")
        return None, None, None
    
    nodes, prediction_edges, cycles = _causal_discovery_pipeline(f'{benchmark_title} - Prediction', '', entities=ground_truth_nodes, use_text_in_causal_discovery=False, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=False, use_text_in_entity_optimization=False, search_cycles=True, plot_graphs=plot_graphs, plot_interactive_graph=False, verbose=verbose)
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


# In[30]:


def precision_recall_curve_plot(curves):
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


# In[31]:


def f1_score_hist(curves):
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


# In[4]:


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

precision_recall_curve_plot(curves)
f1_score_hist(curves)


# In[86]:


get_ipython().system('pip install --upgrade nbformat')


# In[32]:


precision_recall_curve_plot(curves)
f1_score_hist(curves)


# In[54]:


ground_truth_nodes = ['alcohol', 'liver cirrhosis', 'death']
ground_truth_edges = [('alcohol', 'liver cirrhosis'), ('liver cirrhosis', 'death'), ('alcohol', 'death')]
title = 'Alcohol_benchmark'
shd, aupr, curve = benchmark_evaluation(title, ground_truth_nodes, ground_truth_edges, plot_graphs=True, verbose=True)


# ### Intervention Hamming Distance

# In[66]:


# minimal number of interventions (edge addition/removal/flip) to turn graph_1 into graph_2


# ## Asia

# In[24]:


ground_truth_nodes = ['visit to Asia', 'tubercolosis', 'lung cancer', 'bronchitis', 'dyspnoea', 'smoking', 'positive X-ray']
ground_truth_edges = [('visit to Asia', 'tubercolosis'), ('smoking', 'lung cancer'), ('smoking', 'bronchitis'), ('bronchitis', 'dyspnoea'), ('lung cancer', 'dyspnoea'), ('tubercolosis', 'dyspnoea'), ('lung cancer', 'positive X-ray'), ('tubercolosis', 'positive X-ray')]
# plot_interactive_graph(ground_truth_nodes, ground_truth_edges, name='Asia - Ground Truth')
plot_causal_graph(ground_truth_nodes, ground_truth_edges, title='Asia - Ground Truth')


# In[29]:


import time

text = 'Shortness-of-breath (dyspnoea) may be due to tuberculosis, lung cancer or bronchitis, or none of them, or more than one of them. A recent visit to Asia increases the chances of tuberculosis, while smoking is known to be a risk factor for both lung cancer and bronchitis. The results of a single chest X-ray do not discriminate between lung cancer and tuberculosis, as neither does the presence or absence of dyspnoea.'
text_title = f'Asia_{time.time().as_integer_ratio()[0]}'
nodes, edges = causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=True, optimize_found_entities=True, use_text_in_entity_optimization=False, verbose=True)


# In[118]:


import time

text = 'Shortness-of-breath (dyspnoea) may be due to tuberculosis, lung cancer or bronchitis, or none of them, or more than one of them. A recent visit to Asia increases the chances of tuberculosis, while smoking is known to be a risk factor for both lung cancer and bronchitis. The results of a single chest X-ray do not discriminate between lung cancer and tuberculosis, as neither does the presence or absence of dyspnoea.'
text_title = f'Asia_{time.time().as_integer_ratio()[0]}'
nodes, edges = causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=True, optimize_found_entities=True, use_text_in_entity_optimization=False, verbose=True)


# ## Alcohol

# In[28]:


nodes = ['alcohol', 'liver cirrhosis', 'death']
edges = [('alcohol', 'liver cirrhosis'), ('liver cirrhosis', 'death'), ('alcohol', 'death')]
plot_causal_graph(nodes, edges, title='Alcohol - Ground Truth')


# In[24]:


text = 'Excessive alcohol consumption can lead to liver cirrhosis, which can ultimately result in death.'
text_title = 'Alcohol'
causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=True, optimize_found_entities=True, use_text_in_entity_optimization=True, verbose=True)


# ## Cancer

# In[25]:


nodes = ['smoking', 'respiratory disease', 'lung cancer', 'asbestos exposure']
edges = [('smoking', 'respiratory disease'), ('respiratory disease', 'lung cancer'), ('asbestos exposure', 'lung cancer'), ('asbestos exposure', 'respiratory disease'), ('smoking', 'lung cancer')]
plot_causal_graph(nodes, edges, title='Cancer - Ground Truth')


# In[ ]:


text = 'Long-term asbestos exposure and smoking are significant risk factors for developing respiratory diseases, including lung cancer.'
text_title = 'Cancer'
causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=True, optimize_found_entities=True, use_text_in_entity_optimization=True, verbose=True)


# In[27]:


text = 'Long-term asbestos exposure and smoking are significant risk factors for developing respiratory diseases, including lung cancer.'
text_title = 'Cancer'
causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=True, optimize_found_entities=True, use_text_in_entity_optimization=True, verbose=True)


# ## Diabetes

# In[29]:


nodes = ['lack of exercise', 'body weight', 'diabetes', 'diet']
edges = [('lack of exercise', 'body weight'), ('lack of exercise', 'diabetes'), ('body weight', 'diabetes'), ('diet', 'diabetes'), ('diet', 'body weight')]
plot_causal_graph(nodes, edges, title='Diabetes - Ground Truth')


# In[33]:


text = 'Unhealthy diet, lack of exercise, and excessive body weight can contribute to the development of diabetes.'
text_title = 'Diabetes'
causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=True, optimize_found_entities=True, use_text_in_entity_optimization=True, verbose=True)


# In[79]:


text = 'Unhealthy diet, lack of exercise, and excessive body weight can contribute to the development of diabetes.'
text_title = 'Diabetes'
gpt_ner(text)
# causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=True, optimize_found_entities=True, use_text_in_entity_optimization=True, verbose=True)


# ## Obesity

# In[26]:


nodes = ['obesity', 'mortality', 'heart failure', 'heart defects']
edges = [('obesity', 'mortality'), ('obesity', 'heart failure'), ('heart failure', 'mortality'), ('heart defects', 'heart failure'), ('heart defects', 'mortality')]
plot_causal_graph(nodes, edges, title='Obesity - Ground Truth')


# In[87]:


text = 'Obesity is associated with an increased risk of mortality, mainly due to its connection to heart failure and heart defects.'
# text = 'Obesity can cause heart failure, which can lead to mortality. Heart defects can also cause heart failure, which can also lead to mortality. Both heart defects and obesity can lead to mortality.'
text = 'Obesity and heart defects can cause a heart failure, which can lead to mortality. Both heart defects and obesity can also lead to mortality.'
text = 'Obesity and heart defects can cause both mortality and heart failures, which can itself lead to mortality.'
text_title = 'Obesity'
causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=True, optimize_found_entities=False, use_text_in_entity_optimization=True, verbose=True)


# In[38]:


text = 'Obesity is associated with an increased risk of mortality, mainly due to its connection to heart failure and heart defects.'
text = 'Obesity can cause heart failure, which can lead to mortality. Heart defects can also cause heart failure, which can also lead to mortality. Both heart defects and obesity can lead to mortality.'
text_title = 'Obesity'
causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=False, use_text_in_entity_optimization=True, verbose=True)


# # PubMed Causal Discovery

# In[20]:


df = pd.read_csv('../data/pubmed_data.csv') 

text = df['abstract'][0]
text


# In[23]:


sample = df['abstract'].sample().values[0]
print(f'{sample}\n')

causal_discovery_pipeline('test', sample, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=True, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, verbose=True)


# # NOTES

# For example, if we wanted to elicit a prediction for the direction of an edge between variables with metadata µi: “lung cancer”, µj : “cigarette smoking”, and causation verb verb_k: “causes” we would use the following prompt:
# 
# _Among these two options which one is the most likely true: \
# (A) lung cancer causes cigarette smoking \
# (B) cigarette smoking causes lung cancer \
# The answer is:_
# 
# We then compute the log probability of the responses (A) and (B), and use the softmax to obtain a probability distribution
# over the directions of the edge (Kadavath et al., 2022). Since we rely on scoring, instead of generation, the output of the
# LLM-expert is deterministic given a fixed prompt. To foster randomness in the LLM-expert outputs, we randomly draw
# verb_k from the following verbs of causation: provokes, triggers, causes, leads to, induces, results in, brings about, yields,
# generates, initiates, produces, stimulates, instigates, fosters, engenders, promotes, catalyzes, gives rise to, spurs, and sparks.
# 
# introducing variation in the verbs used to express causality in your list of options could potentially lead to more diverse and coherent answers. GPT-3, being deterministic given the same prompt, can sometimes produce more interesting and nuanced responses when provided with varied input. By incorporating synonyms for "causes" in your options, you encourage the model to explore different linguistic expressions of causality, which can lead to more coherent and contextually appropriate responses. using synonyms like "provokes", "triggers", "results in", and others can help in generating more diverse and contextually relevant cause-and-effect relationships. It can potentially lead to a richer exploration of the underlying causal connections between the entities you're asking about.
# 
# verbs = ['provokes', 'triggers', 'causes', 'leads to', 'induces', 'results in', 'brings about', 'yields',
# 'generates', 'initiates', 'produces', 'stimulates', 'instigates', 'fosters', 'engenders', 'promotes', 'catalyzes', 'gives rise to', 'spurs', 'sparks']
# 
# 
# 
# 
# 
# 
# 1. eventually add new benchmarks in code ('Asia'-like graphs from '_Causal Discovery with Language Models as Imperfect Experts_' = https://www.bnlearn.com/bnrepository/ and DAGs in '_progetto diploma/Ground Truth DAGs for medical benchmark.PNG_' from '_Can Large Language Models Build Causal Graphs?_' )
# 2. update docs with benchmarks
# 3. update docs with DFS acyclicity check
# 4. update causal query with random verbs (single or multi query for each pair of variables, then find edge with best score ??)
# 5. ask what to do in case of a cycle (e.g., extract edges that cause it and requery LLM for new edges ??)
# 
# 
# 
# 
# https://arxiv.org/ftp/arxiv/papers/1302/1302.4972.pdf - Causal inference and causal explanation with background 
# knowledge
# 
# 
# # TODO - update docs with benchmarks
# 
# 
# 1. benchmarks
#     - pipeline completa (con NER) solo per grafi che hanno un testo, tutti gli altri skippano step di NER e fanno direttamente Causal Discovery con le entità vere/finali
#     - precision, recall, f1, SHD, IHM dei risultati sulla Ground Truth 
# 
# 2. sinonimi
#     - capire se tenere step ottimizzazione entità
#     - nel caso step è tenuto, associare sinonimi tra loro
# 
# 3. ciclo
#     - testo contraffatto per generare ciclo
#     - ~~evidenziare parte di grafo (nodi e archi) che lo costituiscono~~
#     
# 
#     - DFS/networkX per controllo se c'è ciclo, nel caso rimozione di un arco (pensare a quale:  se 'A --> B --> C --> D --> A'  possibile rimuovere "l'ultimo" arco, D --> A) e ripetere finche grafo diventa aciclico
# 
# 4. grafo interattivo, con hover/click su arco mostra parte di testo che motiva quell'arco
# 
# 5. opzione per forzare GPT a basarsi solo sul testo e non su sue conoscenze pre-trained

# # Questions
# 
# 1. What to do in case of a cyclic graph ? ()
# 2. Evalutation on benchmark ? \
#      a. handle cases of graphs with different nodes, or nodes with different names \
#      b. eiter _**Structural Hamming Distance**_ (differences in missing or reversed edges) or _**Structural Intervention Distance**_ (number of manipulations to turn **graph_1** into **graph_2**)
# 3. Benchmarks without the NER step (would solve problem 2.a of nodes with different names) ? Or with the NER step on a hand-written/generated text starting from the entities and the edges of the GT?
# 4. Random verbs (e.g. 'cause'/'may lead to'/'provokes'/...) in causal discovery query to gpt? 
# 5. In the docs, is it best to move the causality and graph definitions at the very beginning (or at the beginning of the causal analysis chapter)?
# 6. In the docs, is it best to move all the python code snippets in an Appendix section and add references in the text?

# # GPT Prompt engineering for devs
# https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/
# 
# https://artificialcorner.com/openai-and-andrew-ng-just-released-a-free-chatgpt-prompt-engineering-course-b0884c03e946
# 
# https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api

# ## PRINCIPLE 1 : Write clear and specific instructions

# - use delimiters (brackets, xml tags, quotes)
# - ask for structured output
# - check whether conditions are satisfied; check assumptions required to do the task (e.g.: summarize text instructions into a series of steps, otherwise write 'No steps provided')
# - Few-shot prompting: give successful examples of completing tasks, then ask the model to perform the task (e.g.: \
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;''' \
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Answer in a consistent style: \
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<**child**>Teach me about patience \
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<**grandparent**>: The river that carves the deepest valley flows from a modest spring, .... \
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<**child**>Teach me about resilience \
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;''' \
#     )

# ## PRINCIPLE 2 : Give the model time to "think"
# 
# - request a chain or series of relevent reasoning before the model provides the final answer
# - instruct the model to work out its own solution before rushing to conclusions (if the model has to check if a provided solution is correct, ask it to come up with one of his own and compare it with the provided one)

# ## LIMITATIONS
# 
# 1. **Hallucinations** \
# Even though the model has been exposed to vast amounts of textual knowledge during its training process, it has not perfectly memorized the information it has seen: this may bring the model to try answering questions about obscure topics and can make things up that sound plausible but are not actually true, known as hallucinations (e.g.: asking the model to describe a non-existing product of a real company) \
# ==> **To reduce hallucinations**, ask the model to find relevant information, then answer the question based on the relevant information found (try tracing the answer back to the source document).
# 
# 

# ## ITERATIVE PROMPT CRAFTING
# 1. try out an idea
# 2. implementation
# 3. see results
# 4. error analysis
# 
# In general:
# - clarify instructions
# - give more time to think
# - refine prompts with examples

# ## SUMMARIZING
# 
# - word/sentence(/character) limit
# - focus on particular aspects (e.g.: from an online review, focus on product quality and cost/shipment and delivery)
# - **extract** (extracts info from text) vs. **summarize** (summaries may include topics that are not related to the topics of focus) with defined focus
# 

# ## INFERRING
# 
# Analysis on text given as input: extracting names, labels, sentiment analysis,... . \
# In traditional ML workflow, this is done by collecting dataset, training and deploy a model (a separate model for each application: NER, classification, ...); LLMs give a large speed up in application development, by allowing devs to write a prompt and generate results right away for many different tasks.
# 
# E.g.: detecting if a customer is angry in his review (to then make customer support contact him) is achievable with supervised learning but would require a large dataset to train and deploy a model; with LLMs this can be done with a simple (yet sufficiently correct) prompt. \
# 
# **MULTIPLE TASKS AT ONCE**: \
# ~~~
# prompt = f"""
# Identify the following items from the review text: 
# - Sentiment (positive or negative)
# - Is the reviewer expressing anger? (true or false)
# - Item purchased by reviewer
# - Company that made the item
# 
# The review is delimited with triple backticks. \
# Format your response as a JSON object with \
# "Sentiment", "Anger", "Item" and "Brand" as the keys.
# If the information isn't present, use "unknown" \
# as the value.
# Make your response as short as possible.
# Format the Anger value as a boolean.
# 
# Review text: '''{lamp_review}'''
# """
# ~~~
# 
# Infer topics of given articles (e.g.: USA, federal government, NASA, space exploration)
# 

# In[33]:


x = '''{
  "nasa": 1,
  "local government": 0,
  "engineering": 0,
  "employee satisfaction": 1,
  "federal government": 1
}'''
import json
x = json.loads(x)
x


# ## EXPANDING
# 
# taking a small amount of input data (set of instructions or a list of topics) and generate longer pieces of text. \
# 
# **temperature** is the degree of exploration or randomness of the model's output. With temperature=**0**, the LLM will choose only the next most likely word; with higher temperatures, the LLM will choose with a higher chance one of the less likely words. \
# E.g.:   With  "_My favorite food is_" the next most likely words are with 53% chance _pizza_, 30% chance _sushi_ and 5% chance _tacos_; \
# with temp=0, the LLM will ALWAYS choose pizza; with higher temperature, it chooses with a higher chance pizza and tacos.

# ## CHAT WORKFLOW
# 
# **system**: overall instruction, sets behaviour and persona of the model, high level instruction for the conversation
# **assistant**: the model responses
# **user**: the final user
# 
# 
# every request is standalone; to make the model "remember" older conversations (like for building a chatbot), the request must contain all the necessary context information (in the form of _system_, _assistant_ and _user_ messages)
# 
