from django.shortcuts import render, render_to_response, HttpResponse
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
import pandas as pd
import numpy as np
import networkx as nx
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import *


# Create your views here.
def homepage(request):
    G = nx.karate_club_graph()

    plot = Plot(plot_width=400, plot_height=400, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
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

    # Return to django homepage with componenents sent as arguments which will then be displayed
    return render_to_response('pages/base.html', dict(script=script, div=div))

def visualization1(request):
    G = nx.karate_club_graph()

    plot = Plot(plot_width=400, plot_height=400, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
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
    return render_to_response('pages/visualization1.html', dict(script=script, div=div))

def visualization2(request):
    G = nx.karate_club_graph()

    plot = Plot(plot_width=400, plot_height=400, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
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

