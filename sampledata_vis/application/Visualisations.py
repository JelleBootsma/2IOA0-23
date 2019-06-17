import csv
import math
import time
import scipy
import pandas as pd
import numpy as np
import networkx as nx
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, ColumnDataSource, \
    LinearColorMapper
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import *
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from .forms import DataForm
from .models import Data
from bokeh import events
from numpy import pi
from bokeh.models.widgets import Tabs, Panel, Select
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar)
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import colors
import holoviews as hv  # There is a reason we have to do this here but its not important. Holoviews is the next library
from bokeh.transform import linear_cmap
from bokeh.transform import transform
from bokeh.palettes import Spectral4
from bokeh.layouts import column
from scipy.cluster.hierarchy import dendrogram, linkage, ClusterNode, to_tree
from scipy.spatial.distance import squareform
from numpy import arange
from bokehheat import heat
import pandas as pd
import numpy as np
from bokeh.layouts import row, widgetbox, column
from bokeh.models import ColumnDataSource, CustomJS, StaticLayoutProvider, Oval, Circle
from bokeh.models import HoverTool, TapTool, BoxSelectTool, GraphRenderer, MultiLine
from bokeh.models.widgets import RangeSlider, Button, DataTable, TableColumn, NumberFormatter, Select
from bokeh.io import curdoc, show, output_notebook
from bokeh.plotting import figure
import networkx as nx
from bokeh.io import show, output_file
from bokeh.plotting import figure, Column
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes, NodesOnly

hv.extension('bokeh')

def Adjacent(doc):
    args = doc.session_context.request.arguments
    file = args.get('file')[0]
    file = str(file.decode('UTF-8'))

    try:
        df = pd.read_csv("media/" + file, sep=';')
        print('Loaded data succesfully')
    except:
        raise Exception("File does not exist")
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

    # If data too large
    #########################################################
    N = len(names)
    if len(names) > 110:
        counts = np.delete(counts, np.s_[110:N], axis=0)
        counts = np.delete(counts,  np.s_[110:N], axis=1)
    if len(names) > 110:
        names = np.delete(names, np.s_[110:N])

    # Deleting duplicates
    #########################################################
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

    deleted = 0
    index_K = 0
    for k in counts:
        index = np.where(k == 0.0)
        if len(index[0]) == (len(counts) + deleted):
            counts = np.delete(counts, index_K - deleted, axis=0)
            counts = np.delete(counts, index_K - deleted, axis=1)
            names = np.delete(names, index_K - deleted)
            deleted = deleted + 1
        index_K = index_K + 1
    #########################################################

    # Make a distance matrix
    #######################################################
    N = len(counts)
    distancematrix = np.zeros((N, N))
    count = 0
    for node_1 in counts:
        distancematrix[count] = node_1
        count = count + 1

    for m in range(N):
        for n in range(N):
            if distancematrix[m][n] == 0:
                distancematrix[m][n] = float("inf")
    for l in range(N):
        distancematrix[l][l] = 0

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if distancematrix[i][j] > distancematrix[i][k] + distancematrix[k][j]:
                    distancematrix[i][j] = distancematrix[i][k] + distancematrix[k][j]

    # Reorder alphabetically
    ########################################################
    namesAlph = np.array(sorted(names))
    N = len(namesAlph)
    nodesAlph = np.zeros((N, N))
    index_x_2 = 0
    index_y_2 = 0
    for name_x in namesAlph:
        for name_y in namesAlph:
            index_y = np.where(names == name_y)
            index_x = np.where(names == name_x)
            nodesAlph[index_x_2][index_y_2] = counts[index_x[0][0]][index_y[0][0]]
            index_y_2 = index_y_2 + 1
        index_y_2 = 0
        index_x_2 = index_x_2 + 1
    #########################################################

    # Reorder hierarchy for increasingly and decreasing
    ########################################################
    N = len(counts)
    distanceM = np.zeros((N, N))
    distanceM_2 = np.zeros((N, N))
    distanceM_3 = np.zeros((N, N))
    count = 0
    for node in distancematrix:
        distanceM[count] = node
        count = count + 1
    namesHeirRow = [""] * len(names)
    namesHeirColumn = [""] * len(names)

    # SORTING COLUMNS
    sumsOfRows = []
    sum = 0
    index = 0
    for rows in distanceM:
        for value in rows:
            sum = sum + value
        sumsOfRows.append([sum, index])
        sum = 0
        index = index + 1
    sumsOfRows = sorted(sumsOfRows)
    index = 0
    for sum in sumsOfRows:
        for rows in range(0, len(distanceM)):
            distanceM_2[rows][index] = distanceM[rows][sum[1]]
            namesHeirColumn[index] = names[sum[1]]
        index = index + 1

    # SORTING ROWS
    sumsOfRows = []
    sum = 0
    index = 0
    for rows in distanceM_2:
        for value in rows:
            sum = sum + value
        sumsOfRows.append([sum, index])
        sum = 0
        index = index + 1
    sumsOfRows = sorted(sumsOfRows)
    index = 0
    for sum in sumsOfRows:
        distanceM_3[index] = distanceM_2[sum[1]]
        namesHeirRow[index] = names[sum[1]]
        index = index + 1
    #######################################################

    # Establishing all the values
    #######################################################
    xname = []
    yname = []
    alpha = []
    for i, node1 in enumerate(counts):
        for j, node2 in enumerate(counts):
            xname.append(names[i])
            yname.append(names[j])
            alpha.append(min(counts[i][j], 0.6) + 0.3)
    xname_2 = []
    yname_2 = []
    alpha_2 = []
    for i, node1 in enumerate(nodesAlph):
        for j, node2 in enumerate(nodesAlph):
            xname_2.append(namesAlph[i])
            yname_2.append(namesAlph[j])
            alpha_2.append(min(nodesAlph[i][j], 0.6) + 0.3)
    xname_3 = []
    yname_3 = []
    alpha_3 = []
    for i, node1 in enumerate(distanceM_3):
        for j, node2 in enumerate(distanceM_3):
            xname_3.append(namesHeirColumn[i])
            yname_3.append(namesHeirRow[j])
            alpha_3.append(min(distanceM_3[i][j], 0.6) + 0.3)
    #######################################################

    # Creating a color map
    #######################################################
    map = cm.get_cmap("BuPu")
    bokehpalette = [mpl.colors.rgb2hex(m) for m in map(np.arange(map.N))]
    mapper = LinearColorMapper(palette=bokehpalette, low=counts.min().min(), high=counts.max().max())
    mapper_2 = LinearColorMapper(palette=bokehpalette, low=distanceM_3.min().min(), high=(distanceM_3.max().max()))
    ######################################################

    data = dict(
        xname=xname,
        yname=yname,
        alphas=alpha,
        count=counts.flatten(),
        xname_2=xname_2,
        yname_2=yname_2,
        alphas_2=alpha_2,
        count_2=nodesAlph.flatten(),
        xname_3=xname_3,
        yname_3=yname_3,
        alphas_3=alpha_3,
        count_3=distancematrix.flatten(),
        count_4=distanceM_3.flatten()
    )

    # Plot -- default
    #######################################################
    p = figure(x_axis_location="above", tools="hover,save,wheel_zoom,box_zoom,reset",
               y_range=list(reversed(names)), x_range=names,
               tooltips=[('names', '@yname, @xname'), ('count', '@count')])
    p.plot_width = 1200
    p.plot_height = 1000
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 1
    p.xaxis.major_label_orientation = np.pi / 3

    # Plot -- alphabetical
    #######################################################
    p2 = figure(x_axis_location="above", tools="hover,save,wheel_zoom,box_zoom,reset",
                y_range=list(reversed(namesAlph)), x_range=namesAlph,
                tooltips=[('names', '@yname_2, @xname_2'), ('count_2', '@count_2')])
    p2.plot_width = 1200
    p2.plot_height = 1000
    p2.grid.grid_line_color = None
    p2.axis.axis_line_color = None
    p2.axis.major_tick_line_color = None
    p2.axis.major_label_text_font_size = "8pt"
    p2.axis.major_label_standoff = 1
    p2.xaxis.major_label_orientation = np.pi / 3

    # Plot -- distance matrix
    ######################################################
    p3 = figure(x_axis_location="above", tools="hover,save,wheel_zoom,box_zoom,reset",
                y_range=list(reversed(names)), x_range=names,
                tooltips=[('names', '@yname, @xname'), ('count_3', '@count_3')])
    p3.plot_width = 1200
    p3.plot_height = 1000
    p3.grid.grid_line_color = None
    p3.axis.axis_line_color = None
    p3.axis.major_tick_line_color = None
    p3.axis.major_label_text_font_size = "8pt"
    p3.axis.major_label_standoff = 1
    p3.xaxis.major_label_orientation = np.pi / 3

    # Plot -- hierarchy -- increasing
    #######################################################
    p4 = figure(x_axis_location="above", tools="hover,save,wheel_zoom,box_zoom,reset",
                y_range=list(reversed(namesHeirRow)), x_range=namesHeirColumn,
                tooltips=[('names', '@yname_3, @xname_3'), ('count_4', '@count_4')])
    p4.plot_width = 1200
    p4.plot_height = 1000
    p4.grid.grid_line_color = None
    p4.axis.axis_line_color = None
    p4.axis.major_tick_line_color = None
    p4.axis.major_label_text_font_size = "8pt"
    p4.axis.major_label_standoff = 1
    p4.xaxis.major_label_orientation = np.pi / 3
    #######################################################

    # Plot -- hierarchy -- decreasing
    #######################################################
    p5 = figure(x_axis_location="above", tools="hover,save,wheel_zoom,box_zoom,reset",
                y_range=namesHeirRow, x_range=list(reversed(namesHeirColumn)),
                tooltips=[('names', '@yname_3, @xname_3'), ('count_4', '@count_4')])
    p5.plot_width = 1200
    p5.plot_height = 1000
    p5.grid.grid_line_color = None
    p5.axis.axis_line_color = None
    p5.axis.major_tick_line_color = None
    p5.axis.major_label_text_font_size = "8pt"
    p5.axis.major_label_standoff = 1
    p5.xaxis.major_label_orientation = np.pi / 3
    #######################################################

    tab1 = Panel(child=p, title="Adjacency Matrix")
    tab2 = Panel(child=p2, title="Alphabetical Adjacency")
    tab3 = Panel(child=p3, title="Distance Matrix")
    tab4 = Panel(child=p4, title="Increasing Distance")
    tab5 = Panel(child=p5, title="Decreasing Distance")

    tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])

    p.rect('xname', 'yname', 0.9, 0.9, source=data,
           color=transform('count', mapper), alpha='alphas', line_color='#85929E',
           hover_line_color='black', hover_color='black')

    p2.rect('xname_2', 'yname_2', 0.9, 0.9, source=data,
            fill_color=transform('count_2', mapper), alpha='alphas_2', line_color='#85929E',
            hover_line_color='black', hover_color='black')

    p3.rect('xname_2', 'yname_2', 0.9, 0.9, source=data,
            fill_color=transform('count_3', mapper_2), alpha='alphas_3', line_color='#85929E',
            hover_line_color='black', hover_color='black')

    p4.rect('xname_3', 'yname_3', 0.9, 0.9, source=data,
            fill_color=transform('count_4', mapper_2), alpha='alphas_3', line_color='#85929E',
            hover_line_color='black', hover_color='black')

    p5.rect('xname_3', 'yname_3', 0.9, 0.9, source=data,
            fill_color=transform('count_4', mapper_2), alpha='alphas_3', line_color='#85929E',
            hover_line_color='black', hover_color='black')

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10pt",
                         ticker=BasicTicker(desired_num_ticks=1),
                         formatter=PrintfTickFormatter(format="%d"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    p2.add_layout(color_bar, 'right')
    p3.add_layout(color_bar, 'right')
    p4.add_layout(color_bar, 'right')
    p5.add_layout(color_bar, 'right')

    doc.add_root(row(tabs))


def Weighted(doc):
    args = doc.session_context.request.arguments
    file = args.get('file')[0]
    file = str(file.decode('UTF-8'))

    try:
        dfread = pd.read_csv("media/" + file, sep=';')
        print('Loaded data succesfully')
    except:
        raise Exception("File does not exist")

    dfSort = dfread.sort_index()
    nArr = dfread.index.values
    dfArr = dfread.values
    dfnsort = pd.DataFrame(nArr, columns=['names'])
    df = dfnsort.sort_values(by=['names'])
    nArrSort = dfSort.index.values
    nArrSortND = np.asarray(list(dict.fromkeys(nArrSort)))
    dfArrSort = dfSort.values

    # Import / instantiate networkx graph
    G = nx.Graph()

    for x in range(0, len(df) - 1):
        xVal = x + 1
        for y in range(0, x):
            if dfArr[xVal][y] > 0.0:
                G.add_edge(nArr[xVal], nArr[y], weight=dfArr[xVal][y])

    # Node Characteristics
    name = nArrSortND[0]
    # node_name = list(G.nodes())
    # positions = nx.circular_layout(G)

    # size = [3 for k in range(len(G.nodes()))]
    # nx.set_node_attributes(G, size, 'size')
    # visual_attributes=ColumnDataSource(
    #    pd.DataFrame.from_dict({k:v for k,v in G.nodes(data=True)},orient='index'))

    # Edge characteristics
    start_edge = [start_edge for (start_edge, end_edge) in G.edges()]
    end_edge = [end_edge for (start_edge, end_edge) in G.edges()]
    weight = list(nx.get_edge_attributes(G, 'weight').values())

    edge_df = pd.DataFrame({'source': start_edge, 'target': end_edge, 'weight': weight})

    # Create full graph from edgelist
    G = nx.from_pandas_edgelist(edge_df, edge_attr=True)

    # Convert full graph to Bokeh network for node coordinates and instantiate Bokeh graph object
    G_source = from_networkx(G, nx.circular_layout, scale=2, center=(0, 0))
    graph = GraphRenderer()

    # Update loop where the magic happens
    def update():
        node_name_ex = np.delete(nArrSortND, np.where(nArrSortND == select.value))
        name = select.value
        namenumber = 0
        for x in range(len(df)):
            if nArr[x] == select.value:
                namenumber = x
        selected_array = []
        for i in range(len(edge_df)):
            if edge_df['source'][i] == select.value or edge_df['target'][i] == select.value:
                selected_array.append(edge_df.loc[[i]])
        selected_df = pd.concat(selected_array)
        sub_G = nx.from_pandas_edgelist(selected_df, edge_attr=True)
        subArr = selected_df.index.values
        subArrVal = selected_df.values
        tempArr = node_name_ex

        size = []
        alpha = []
        alpha_hover = []
        width = []
        edgeName = []
        line_color = []
        i = 0

        for p in range(len(nArrSortND)):
            if nArrSortND[p] == select.value:
                size.append(25)
            else:
                if nArrSortND[p] in subArrVal:
                    if selected_df['source'][subArr[i]] == select.value:
                        # sub_G.node[selected_df['target'][subArr[i]]]['size'] = 25 * sub_G[select.value][selected_df['target'][subArr[i]]]['weight']
                        tempArr = np.delete(tempArr, np.where(tempArr == selected_df['target'][subArr[i]]))
                        alpha.append(sub_G[selected_df['target'][subArr[i]]][select.value]['weight'])
                        alpha_hover.append(1)
                        width.append(int(5 * sub_G[selected_df['target'][subArr[i]]][select.value]['weight']))
                        line_color.append('#FF0000')
                        edgeName.append([select.value, nArrSortND[p]])
                        size.append(25 * sub_G[select.value][selected_df['target'][subArr[i]]]['weight'])
                    else:
                        # sub_G.node[selected_df['source'][subArr[i]]]['size'] = 25 * sub_G[select.value][selected_df['source'][subArr[i]]]['weight']
                        tempArr = np.delete(tempArr, np.where(tempArr == selected_df['source'][subArr[i]]))
                        alpha.append(sub_G[selected_df['source'][subArr[i]]][select.value]['weight'])
                        alpha_hover.append(1)
                        width.append(int(5 * sub_G[selected_df['source'][subArr[i]]][select.value]['weight']))
                        line_color.append('#FF0000')
                        edgeName.append([select.value, nArrSortND[p]])
                        size.append(25 * sub_G[select.value][selected_df['source'][subArr[i]]]['weight'])
                    i = i + 1
                else:
                    sub_G.add_node(nArrSortND[p])
                    # sub_G.node[tempArr[i]]['size'] = 0
                    alpha.append(0)
                    alpha_hover.append(1)
                    width.append(0)
                    line_color.append('#FF0000')
                    edgeName.append(["source", "target"])
                    size.append(0)

        sub_graph = from_networkx(sub_G, nx.circular_layout, scale=2, center=(0, 0))

        tempnArrSortND1 = nArrSortND[(np.where(nArrSortND == select.value)[0][0]):]
        tempnArrSortND2 = nArrSortND[:(np.where(nArrSortND == select.value)[0][0])]

        newnArrSortND = np.append(tempnArrSortND1, tempnArrSortND2)

        nodeSource = ColumnDataSource(data=dict(
            size=size,
            index=nArrSortND,
        ))

        edgeSource = ColumnDataSource(data=dict(
            line_alpha=alpha,
            line_alpha_hover=alpha_hover,
            line_width=width,
            line_color=line_color,
            index=edgeName,
            start=[select.value] * (len(nArrSortND) - 1),
            end=newnArrSortND[1:],
        ))
        # print('node data:')
        # print('size; ', len(size))
        # print('node names; ', len(nArrSortND))
        #
        # print('edge data:')
        # print('alpha; ', len(alpha))
        # print('alpha hover; ', len(alpha_hover))
        # print('width; ', len(width))
        # print('color; ', len(line_color))
        # print('edge names; ', len(edgeName))
        # print('start; ', len([nArrSortND[0]] * (len(nArrSortND) - 1)))
        # print('end; ', len(nArrSortND[1:]))

        sub_graph.edge_renderer.data_source = edgeSource
        sub_graph.node_renderer.data_source = nodeSource

        graph.edge_renderer.data_source.data = sub_graph.edge_renderer.data_source.data
        graph.node_renderer.data_source.data = sub_graph.node_renderer.data_source.data
        graph.node_renderer.data_source.add(size, 'size')

    def selected_points(attr, old, new):
        selected_idx = graph.node_renderer.selected.indices  # does not work
        #print(selected_idx)

    # Slider which changes values to update the graph
    select = Select(title='names', options=nArrSortND.tolist(), value=nArrSortND[0])
    select.on_change('value', lambda attr, old, new: update())

    positions = nx.circular_layout(G)

    # Plot object which is updated
    plot = figure(title="Meetup Network Analysis", x_range=(-2.2, 2.2), y_range=(-1.1, 1.1),
                  tools="pan,wheel_zoom,box_select,reset,box_zoom", plot_width=1400, plot_height=700)

    # Assign layout for nodes, render graph, and add hover tool
    graph.layout_provider = StaticLayoutProvider(graph_layout=positions)

    graph.node_renderer.glyph = Circle(size='size', fill_color="#FCFCFC")
    graph.node_renderer.selection_glyph = Circle(size='size', fill_color="#000000")
    graph.node_renderer.hover_glyph = Circle(size='size', fill_color="#22A784")

    graph.edge_renderer.glyph = MultiLine(line_color='line_color', line_alpha='line_alpha', line_width='line_width')
    graph.edge_renderer.selection_glyph = MultiLine(line_color='line_color', line_alpha='line_alpha_hover',
                                                    line_width='line_width')
    graph.edge_renderer.hover_glyph = MultiLine(line_color='line_color', line_alpha='line_alpha_hover',
                                                line_width='line_width')

    graph.inspection_policy = NodesAndLinkedEdges and EdgesAndLinkedNodes()
    graph.selection_policy = NodesAndLinkedEdges()
    plot.renderers.append(graph)
    plot.tools.append(HoverTool(tooltips=[("index", "@index"), ("weight", "@line_alpha")]))

    # Set layout
    layout = column(select, plot)

    # does not work
    graph.node_renderer.data_source.selected.on_change("indices", selected_points)

    # Create Bokeh server object
    doc.add_root(layout)
    update()


def Grouped(doc):
    args = doc.session_context.request.arguments
    file = args.get('file')[0]
    file = str(file.decode('UTF-8'))

    with open("media/" + file) as data:
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
            ("Name", "@index")]
        plot = figure(title="", x_range=(-2.2, 2.2), y_range=(-1.1, 1.1),
                      tooltips=TOOLTIPS, plot_width=1400, plot_height=700)
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


def Hierarchical(doc):
    global source, nodes

    """"     # df = pd.read_csv('application/dataSet/GephiMatrix_author_similarity.csv', sep=';')
    #csv_reader = pd.read_csv('application/dataSet/authors.csv', sep=';')

    #############################################################
    # Make a condensed distance matrix
    ############################################################

    # df_std = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    # df_scaled = df_std * (1.0 - 0.0) + 0.0
    #
    # dist = scipy.spatial.distance.squareform(distancematrix)
    # linkage_matrix = linkage(dist, "single")
    # results = dendrogram(linkage_matrix, no_plot=True)
    # icoord, dcoord = results['icoord'], results['dcoord']
    # labels = list(map(int, results['ivl']))
    # df = df.iloc[labels]
    # df_scaled = df_scaled.iloc[labels]
    #
    # tms = []
    #
    #
    # icoord = pd.DataFrame(icoord)

    args = doc.session_context.request.arguments
    print(args)
    file = args.get('file')[0]
    file = str(file.decode('UTF-8'))

    with open("media/" + file) as data:
        csv_reader = csv.reader(data, delimiter=';')

        nArr = csv_reader.index.values
        dfArr = csv_reader.values

        nodes = dfArr
        names = nArr

        N = len(names)
        counts = np.zeros((N, N))
        for i in range(0, len(nodes)):
            for j in range(0, len(nodes)):
                counts[i, j] = nodes[j][i]
                counts[j, i] = nodes[j][i]

        N = len(counts)

        distancematrix = np.zeros((N, N))
        count = 0
        for node_1 in counts:
            distancematrix[count] = node_1
            count = count + 1

        for m in range(N):
            for n in range(N):
                if distancematrix[m][n] == 0:
                    distancematrix[m][n] = float("inf")
        for l in range(N):
            distancematrix[l][l] = 0

        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if distancematrix[i][j] > distancematrix[i][k] + distancematrix[k][j]:
                        distancematrix[i][j] = distancematrix[i][k] + distancematrix[k][j]

        values = distancematrix """
    #########################################################################################################

    def getLevelInfo(tree):
        nodes = [tree.get_left(), tree.get_right()]
        total_desc = tree.get_count()
        percents = [0]
        names = []
        for node in nodes:
            percentage = float(node.get_count()) / float(total_desc)
            percents.append(float(percentage + percents[-1]))
            names.append(node.get_id())

        return percents, names, nodes

    def genDataSource(tree):
        percents, names, nodes = getLevelInfo(tree)

        # define starts/ends for wedges from percentages of a circle
        starts = [p * 2 * pi for p in percents[:-1]]
        ends = [p * 2 * pi for p in percents[1:]]
        colours = getColours(len(starts))
        branchLengths = [node.dist for node in nodes]
        children = [node.get_count() for node in nodes]
        source = ColumnDataSource(data=dict(
            start=starts, end=ends, name=names, colour=colours, branchLength=branchLengths, children=children
        ))
        return source, nodes

    def getColours(Length):
        colours = ["red", "green", "blue", "orange", "yellow", "purple", "pink"]
        returnColours = colours
        while len(returnColours) <= Length:
            returnColours += colours
        if returnColours[-1] == "red":
            returnColours[-1] = "orange"

        return returnColours[0:Length]

    def calcAngle(x, y):
        innerProduct = x
        lengthProduct = math.sqrt(x ** 2 + y ** 2)
        cosAngle = innerProduct / lengthProduct
        if y < 0 and x > 0:
            return 2 * pi - math.acos(cosAngle)
        else:
            return math.acos(cosAngle)

    def update(event):
        print('Click registered')
        angle = calcAngle(event.x, event.y)
        print(angle)
        global source, nodes
        for i in range(len(source.data['end'])):
            if source.data['end'][i] > angle and source.data['start'][i] < angle:
                clickedNode = i
                print(i)

        if nodes[clickedNode].get_count() > 2:
            new_source, nodes = genDataSource(nodes[clickedNode])
            source.data = new_source.data


    def returnVisualisation():
        global source, nodes
        new_source, nodes = genDataSource(tree)
        source.data = new_source.data

    args = doc.session_context.request.arguments
    file = args.get('file')[0]
    file = str(file.decode('UTF-8'))

    try:
        df = pd.read_csv("media/" + file, sep=';')
        print('Loaded data succesfully')
    except:
        raise Exception("File does not exist")

    names = df.index.values
    counts = df.values

    # If data too large
    #########################################################
    if len(names) > 50:
        n = 50
        while len(names) != 50:
            names = np.delete(names, (n))
            counts = np.delete(counts, (n), axis=0)
            counts = np.delete(counts, (n), axis=1)

    counts = np.delete(counts, len(counts), axis = 1)
    # Make a distance matrix
    #######################################################
    N = len(counts)
    distancematrix = np.zeros((N, N))
    count = 0
    for node_1 in counts:
        distancematrix[count] = node_1
        count = count + 1

    for m in range(N):
        for n in range(N):
            if distancematrix[m][n] == 0:
                distancematrix[m][n] = float("inf")
    for l in range(N):
        distancematrix[l][l] = 0

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if distancematrix[i][j] > distancematrix[i][k] + distancematrix[k][j]:
                    distancematrix[i][j] = distancematrix[i][k] + distancematrix[k][j]

    X = distancematrix

    Z = linkage(X, 'ward')

    tree = to_tree(Z)

    ## Create the first data source for the root view
    source, nodes = genDataSource(tree)

    ## Create buttons and tools to interact with the visualisation
    returnButton = Button(label="Return")
    hover = HoverTool()
    hover.tooltips = [("Name", "@name"), ("Lenght to parent", "@branchLength"), ("Children", "@children")]
    hover.mode = 'mouse'
    tools = [hover, 'save']

    ## Create the canvas
    p = figure(x_range=(-1, 1), y_range=(-1, 1), tools=tools)

    ## Draw the wedges on the canvas according to the tree info
    p.wedge(x=0, y=0, radius=1, start_angle='start', end_angle='end', color='colour', alpha=0.6, source=source)

    ## Map actions to events for the interaction
    p.on_event(events.Tap, update)
    returnButton.on_click(returnVisualisation)

    ## Display the visualisation
    doc.add_root(Column(returnButton, p))