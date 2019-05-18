from typing import TextIO

from django.shortcuts import render, render_to_response, HttpResponse
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
import csv
import time
import pandas as pd
import numpy as np
import networkx as nx
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import *


# Create your views here.
def homepage(request):
    return render(request, 'pages/base.html')


def coauthorship(request):
    with open("application/dataSet/GephiMatrix_co-authorship.csv") as data:
        start = time.time()
        csv_reader = csv.reader(data, delimiter=';')
        processedData = list(csv_reader)
        header = processedData[0]
        g = nx.Graph()
        for i in range(1, len(processedData[0])):
            g.add_node(processedData[0][i])
        for i in range(1, len(processedData)):
            for e in range(i, len(processedData[i])):
                try:
                    value = int(processedData[i][e])
                    if value > 0:
                        g.add_edge(header[i], header[e], weight=value)
                except:
                    pass

        TOOLTIPS = [
            ("index", "@index")]
        plot = figure(title="", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tooltips=TOOLTIPS)
        graph = from_networkx(g, nx.spring_layout, scale=2, center=(0, 0))

        graph.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_width=2)
        graph.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=2)
        graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=2)

        graph.node_renderer.glyph = Circle(fill_color=Spectral4[0])
        graph.node_renderer.selection_glyph = Circle(fill_color=Spectral4[2])
        graph.node_renderer.hover_glyph = Circle(fill_color=Spectral4[1])

        graph.selection_policy = NodesAndLinkedEdges()
        graph.inspection_policy = NodesAndLinkedEdges()
        plot.renderers.append(graph)

    # store comments
    script, div = components(plot)
    return render_to_response('pages/visualization1.html', dict(script=script, div=div))


def weightedgraph(request):
    df = pd.read_csv('GephiMatrix_author_similarity.csv', sep=';')
    nArr = df.index.values
    dfArr = df.values

    G = nx.Graph()
    G.add_nodes_from(nArr)

    for x in range(len(df) - 1):
        xVal = x + 1
        for y in range(x):
            if dfArr[xVal][y] > 0.0:
                G.add_edge(nArr[xVal], nArr[y], weight=dfArr[xVal][y])

    plot = Plot(plot_width=600, plot_height=600, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    plot.background_fill_color = "#050976"
    plot.background_fill_alpha = 0
    plot.border_fill_color = "#050976"
    plot.border_fill_alpha = 0

    plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())

    graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0, 0))

    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="#FCFCFC")
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color="#FCFCFC")
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color="#22A784")

    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#FCFCFC", line_alpha=0.8, line_width=5)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color="#FCFCFC", line_width=5)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color="#050976", line_width=5)

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    plot.renderers.append(graph_renderer)

    # store comments
    script, div = components(plot)
    return render_to_response('pages/visualization2.html', dict(script=script, div=div))


def faq(request):
    return render(request, 'pages/FAQ.html')


def data(request):
    return render(request, 'pages/data.html')

def step1(request):
    return render(request, 'pages/step1.html')

def adjacencymatrix(request):
    with open("application/dataSet/GephiMatrix_co-authorship.csv") as data:
        start = time.time()
        csv_reader = csv.reader(data, delimiter=';')
        processedData = list(csv_reader)
        header = processedData[0]
        g = nx.Graph()

        for source, targets in processedData:
            for inner_dict in targets:
                assert len(inner_dict) == 1
                g.add_edge(int(source) - 1, int(inner_dict.keys()[0]) - 1,
                                   weight=inner_dict.values()[0])
        adjacency_matrix = nx.adjacency_matrix(g)

        TOOLTIPS = [
            ("index", "@index")]
        plot = figure(title="", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tooltips=TOOLTIPS)
        graph = from_networkx(g, nx.spring_layout, scale=2, center=(0, 0))

        graph.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_width=2)
        graph.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=2)
        graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=2)

        graph.node_renderer.glyph = Circle(fill_color=Spectral4[0])
        graph.node_renderer.selection_glyph = Circle(fill_color=Spectral4[2])
        graph.node_renderer.hover_glyph = Circle(fill_color=Spectral4[1])

        graph.selection_policy = NodesAndLinkedEdges()
        graph.inspection_policy = NodesAndLinkedEdges()
        plot.renderers.append(graph)

    # store comments
    script, div = components(plot)
    return render_to_response('pages/visualization1.html', dict(script=script, div=div))
