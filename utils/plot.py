"""
This file will contain the functions to plot the results of the experiments
"""

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
import plotly.graph_objects as go
import numpy as np
import pandas as pd 

def plot_graph(graph, node_color=None) -> plt.Figure:
    """
    Plots the graph using the networkx library.
    Args:
        graph: the graph to plot
        node_color: the color of the nodes
    Returns:
        fig: the figure of the graph
    """
    G = to_networkx(graph, to_undirected=True)

    # from https://stackoverflow.com/questions/14283341/how-to-increase-node-spacing-for-networkx-spring-layout
    df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    for row, data in nx.shortest_path_length(G):
        for col, dist in data.items():
            df.loc[row, col] = dist

    df = df.fillna(df.max().max())

    pos = nx.kamada_kawai_layout(G, dist=df.to_dict(), scale=0.5)

    # add positions to the graph
    # pos = nx.spring_layout(G, k=2/np.sqrt(num_nodes), iterations=100, seed=42)
    nx.set_node_attributes(G, pos, "pos")

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_color,  # Display gestational week on hover
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Gestational Weeks',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    return fig