import numpy as np
import gpt
import networkx as nx
import plotly as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from cdt.metrics import SHD, precision_recall
import os


# def plot_causal_graph(nodes, edges, title, paht):

#     if not nodes:
#         return None
    
#     # Create a graph
#     G = nx.DiGraph()

#     # Add nodes
#     for node in nodes:
#         G.add_node(node)

#     # Add edges
#     for e1, e2 in edges:
#         G.add_edge(e1, e2)

#     # Plot the graph
#     pos = nx.spring_layout(G)
#     nx.draw_networkx_nodes(G, pos)
#     nx.draw_networkx_edges(G, pos, arrows=True)
#     nx.draw_networkx_labels(G, pos)
#     #add title to graph
#     plt.title(title)
#     plt.savefig(title, format="PNG")


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def benchmark_evaluation(benchmark_title, ground_truth_nodes, ground_truth_edges, SHD_double_for_anticausal=True, save_graphs=True, graphs_directory='../graphs', verbose=False):

    if ground_truth_nodes is None or ground_truth_edges is None:
        print("Ground truth nodes or edges are None.")
        return None, None, None

    nodes, prediction_edges, cycles = gpt.causal_discovery_pipeline(f'{benchmark_title} - Prediction', '', entities=ground_truth_nodes, use_text_in_causal_discovery=False, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=False, use_text_in_entity_optimization=False, search_cycles=True, plot_static_graph=False, graph_directory_name=graphs_directory, verbose=False)
    
    if verbose:
        print(prediction_edges)

    ground_truth_graph = nx.DiGraph()
    ground_truth_graph.add_nodes_from(nodes)
    ground_truth_graph.add_edges_from(ground_truth_edges)

    prediction_graph = nx.DiGraph()
    prediction_graph.add_nodes_from(nodes)
    prediction_graph.add_edges_from(prediction_edges)

    shd = SHD(ground_truth_graph, prediction_graph, double_for_anticausal=SHD_double_for_anticausal)

    aupr, curve = precision_recall(ground_truth_graph, prediction_graph)

    return shd, aupr, curve, prediction_edges, cycles


def precision_recall_curve_plot(titles, curves, graph_path):
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

    fig.write_html(f'{graph_path}/precision_recall_curve.html')



def f1_score_hist(titles, curves, graph_path):
    fig = go.Figure()

    best_points = [points[1] for points in curves]
    f1s = []
    for i, (precision, recall) in enumerate(best_points):
        f1 = f1_score(precision, recall)
        f1s.append(f1)
        fig.add_trace(go.Bar(
            y=[titles[i]],
            x=[f1],
            orientation='h',
            text=f'{f1:.2f}',
            textposition='inside',
            hoverinfo='x',
            name=titles[i]
        ))

    avg = np.mean(f1s)
    fig.add_trace(go.Bar(
            y=['Average'],
            x=[avg],
            orientation='h',
            text=f'{avg:.2f}',
            textposition='inside',
            hoverinfo='x',
            name='Average'
    ))

    fig.update_layout(
        title='F1 Scores for Benchmarks',
        xaxis_title='F1 Score',
        yaxis_title='Benchmark',
        xaxis=dict(range=[0, 1.1]),
        yaxis=dict(categoryorder='total ascending'),
    )

    fig.write_html(f'{graph_path}/f1_scores.html')


def shd_hist(shd_values, benchmark_titles, graph_path):
    
    shds = shd_values.copy()
    titles = benchmark_titles.copy()

    shds.append(np.mean(shds))
    avg_title = 'Average'
    benchmark_titles.append(avg_title)

    sorted_shds, sorted_titles = zip(*sorted(zip(shds, titles)))

    fig = px.bar(x=sorted_titles, y=sorted_shds, title='Structural Hamming Distance for benchmarks')

    fig.update_xaxes(title_text='Benchmarks')
    fig.update_yaxes(title_text='Structural Hamming Distance')
    fig.write_html(f'{graph_path}/shd_scores.html')


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
    shds = []
    auprs = []
    curves = []
    pred_edges = []
    pred_cycles = []


    main_directory_path = f'../benchmarks/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    graphs_directory_path = f'{main_directory_path}/graphs'
    os.makedirs(graphs_directory_path, exist_ok=True)

    for title, ground_truth_nodes, ground_truth_edges in ground_truth_graphs:
        shd, aupr, curve, prediction_edges, prediction_cycles = benchmark_evaluation(title, ground_truth_nodes, ground_truth_edges, save_graphs=True, graphs_directory=graphs_directory_path, verbose=True)
        titles.append(title)
        shds.append(shd)
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

    precision_recall_curve_plot(titles, curves, main_directory_path)
    f1_score_hist(titles, curves, main_directory_path)
    shd_hist(shds, titles, main_directory_path)
