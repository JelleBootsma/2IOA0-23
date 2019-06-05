from django.shortcuts import render, render_to_response, redirect, HttpResponse
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.layouts import column
import csv
import time
import pandas as pd
import numpy as np
import networkx as nx
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, ColumnDataSource, LinearColorMapper
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import Spectral4

def Adjacent(doc):
    # df = pd.read_csv('application/dataSet/GephiMatrix_author_similarity.csv', sep=';')
    df = pd.read_csv('application/dataSet/authors.csv', sep=';')
    # df = pd.read_csv('application/dataSet/authors_2.csv', sep=';')
    nArr = df.index.values
    dfArr = df.values

    nodes = dfArr
    names = nArr

    N = len(names)
    counts = np.zeros((N, N))
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes)):
            counts[i, j] = nodes[j][i]
            counts[j, i] = nodes[j][i]
    colormap = ["#FDFEFE", "#FCF3CF", "#F9E79F", "#F4D03F", "#F39C12", "#E67E22",
                "#D35400", "#CB4335", "#C0392B", "#7B241C", "#4A235A"]
    arrayi = []
    arrayj = []
    for i in names:
        for j in names:
            indexi = np.where(names == i)
            indexj = np.where(names == j)
            for q in indexi[0]:
                for l in indexj[0]:
                    if i == j and q != l:
                        if q not in arrayj or l not in arrayi:
                            names = np.delete(names, l)
                            arrayi.append(q)
                            arrayj.append(l)

    for j in arrayj:
        counts = np.delete(counts, (j), axis=0)
        counts = np.delete(counts, (j), axis=1)

    # Reorder alphabetically
    ####################################
    alphabetical = "yes"

    if alphabetical == "yes":
        namesOrdered = np.array(sorted(names))
        N = len(namesOrdered)
        nodesOrdered = np.zeros((N, N))
        index_x_2 = 0
        index_y_2 = 0
        for name_x in namesOrdered:
            for name_y in namesOrdered:
                index_y = np.where(names == name_y)
                index_x = np.where(names == name_x)
                nodesOrdered[index_x_2][index_y_2] = counts[index_x[0][0]][index_y[0][0]]
                index_y_2 = index_y_2 + 1
            index_y_2 = 0
            index_x_2 = index_x_2 + 1
        names = namesOrdered
        counts = nodesOrdered
    #####################################

    xname = []
    yname = []
    color = []
    alpha = []
    for i, node1 in enumerate(counts):
        for j, node2 in enumerate(counts):
            xname.append(names[i])
            yname.append(names[j])
            alpha.append(min(counts[i][j], 0.6) + 0.3)
            if counts[i][j] == 0:
                color.append(colormap[0])
            elif counts[i][j] >= 0.5:
                if counts[i][j] > 0.6:
                    if counts[i][j] > 0.7:
                        if counts[i][j] > 0.8:
                            if counts[i][j] > 0.9:
                                if counts[i][j] == 1.0:
                                    color.append(colormap[10])
                                else:
                                    color.append(colormap[9])
                            else:
                                color.append(colormap[8])
                        else:
                            color.append(colormap[7])
                    else:
                        color.append(colormap[6])
                else:
                    color.append(colormap[5])
            else:
                if counts[i][j] < 0.4:
                    if counts[i][j] < 0.3:
                        if counts[i][j] < 0.2:
                            if counts[i][j] < 0.1:
                                color.append(colormap[1])
                            else:
                                color.append(colormap[2])
                        else:
                            color.append(colormap[3])
                    else:
                        color.append(colormap[4])
                else:
                    color.append(colormap[4])
    data = dict(
        xname=xname,
        yname=yname,
        colors=color,
        alphas=alpha,
        count=counts.flatten(),
    )
    p = figure(x_axis_location="above", tools="hover,save,wheel_zoom,box_zoom,reset",
               y_range=list(reversed(names)), x_range=names,
               tooltips=[('names', '@yname, @xname'), ('count', '@count')])
    p.plot_width = 800
    p.plot_height = 800
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 1
    p.xaxis.major_label_orientation = np.pi / 3

    tab1 = Panel(child=p, title="default")
    tab2 = Panel(child=p, title="hierarchical")
    tab3 = Panel(child=p, title="alphabetical")

    tabs = Tabs(tabs=[tab1, tab2, tab3])

    p.rect('xname', 'yname', 0.9, 0.9, source=data,
           color='colors', alpha='alphas', line_color='#85929E',
           hover_line_color='black', hover_color='colors')

    doc.add_root(column(tabs))

def Weighted(doc):


    name = 'Ken_Pier'

    df = pd.read_csv('GephiMatrix_author_similarity.csv', sep=';')

    Graph = nx.Graph()

    plot = Plot(plot_width=600, plot_height=600, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

    plot.background_fill_color = "#FCFCFC"
    plot.background_fill_alpha = 1
    plot.border_fill_color = "#FCFCFC"
    plot.border_fill_alpha = 1

    plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())

    nArr = df.index.values
    dfArr = df.values
    namenumber = 0

    for x in range(len(df)):
        if nArr[x] == name:
            namenumber = x
            break

    node_sizes = []
    alpha = []
    alpha_hover = []
    width = []
    names = []
    edgeName = []
    line_color = []

    numberOfNodes = 1

    Graph.add_node(name)
    node_sizes.append(25)
    names.append(nArr[x])

    for x in range(0, len(df)):
        if 0.0 < dfArr[namenumber][x] and nArr[x] != name:
            Graph.add_node(nArr[x])
            Graph.add_edge(name, nArr[x], weight=dfArr[namenumber][x])
            node_sizes.append(25 * Graph[name][nArr[x]]['weight'])
            alpha.append(Graph[name][nArr[x]]['weight'])
            alpha_hover.append(1)
            width.append(int(5 * Graph[name][nArr[x]]['weight']))
            names.append(nArr[x])
            edgeName.append([nArr[x], name])
            numberOfNodes += 1
            line_color.append('#FF0000')

    # source = ColumnDataSource(pd.DataFrame.from_dict({k:v for k,v in G.nodes(data=True)}, orient='index'))

    nodeSource = ColumnDataSource(data=dict(
        size=node_sizes,
        index=names,
    ))

    edgeSource = ColumnDataSource(data=dict(
        line_alpha=alpha,
        line_alpha_hover=alpha_hover,
        line_width=width,
        line_color=line_color,
        index=edgeName,
        start=[name] * (numberOfNodes - 1),
        end=names[1:],
    ))

    graph_renderer = from_networkx(Graph, nx.shell_layout, center=(0, 0))

    graph_renderer.node_renderer.data_source = nodeSource
    graph_renderer.node_renderer.glyph = Circle(size='size', fill_color="#FCFCFC")
    graph_renderer.node_renderer.selection_glyph = Circle(size='size', fill_color="#000000")
    graph_renderer.node_renderer.hover_glyph = Circle(size='size', fill_color="#22A784")

    graph_renderer.edge_renderer.data_source = edgeSource
    graph_renderer.edge_renderer.glyph = MultiLine(line_color='line_color', line_alpha='line_alpha',
                                                   line_width='line_width')

    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color='line_color', line_alpha='line_alpha_hover',
                                                             line_width='line_width')

    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color='line_color', line_alpha='line_alpha_hover',
                                                         line_width='line_width')

    graph_renderer.selection_policy = NodesAndLinkedEdges() and EdgesAndLinkedNodes()
    graph_renderer.inspection_policy = NodesAndLinkedEdges() and EdgesAndLinkedNodes()

    plot.renderers.append(graph_renderer)
    doc.add_root(column(plot))


def Grouped(doc):
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
        doc.add_root(column(plot))



