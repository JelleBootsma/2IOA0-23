from typing import TextIO, List, Any

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

    df = pd.read_csv('application/dataSet/authors.csv', sep=';')
    nArr = df.index.values
    dfArr = df.values

    print(dfArr)
    nodes = dfArr
    names = nArr

    N = len(names)
    counts = np.zeros((N, N))
    for i in range(0,len(nodes)):
        for j in range(0,len(nodes)):
            counts[i, j] = nodes[j][i]
            counts[j, i] = nodes[j][i]
    colormap = ["#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
                "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]
    xname = []
    yname = []
    color = []
    alpha = []
    for i, node1 in enumerate(counts):
        for j, node2 in enumerate(counts):
            xname.append(names[i])
            yname.append(names[j])
            alpha.append(min(counts[i, j] / 4.0, 0.9) + 0.1)
            if i == j :
                color.append(colormap[1])
            else:
                color.append('lightgrey')
    print('xname', len(xname))
    print('yname', len(yname))
    print('names', len(names))

    data = dict(
        xname=xname,
        yname=yname,
        colors=color,
        alphas=alpha,
        count=counts.flatten(),
    )
    p = figure(title="Les Mis Occurrences",
               x_axis_location="above", tools="hover,save",
               x_range=list(reversed(names)), y_range=names,
               tooltips=[('names', '@yname, @xname'), ('count', '@count')])
    p.plot_width = 800
    p.plot_height = 800
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect('xname', 'yname', 0.9, 0.9, source=data,
           color='colors', alpha='alphas', line_color=None,
           hover_line_color='black', hover_color='colors')

    # store comments
    script, div = components(p)
    return render_to_response('pages/visualization1.html', dict(script=script, div=div))
