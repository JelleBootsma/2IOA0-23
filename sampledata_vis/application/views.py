from typing import TextIO, List, Any
from django import forms
from django.shortcuts import render, render_to_response, redirect, HttpResponse
from bokeh.plotting import figure, output_file, show
from bokeh.embed import server_document
import csv
import time
import pandas as pd
import numpy as np
import networkx as nx
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, ColumnDataSource, LinearColorMapper
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import *
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from .forms import DataForm
from .models import Data



# Create your views here.
def homepage(request):
    return render(request, 'pages/base.html')


def coauthorship(request):
    fileLocation = request.COOKIES['Data']

    # store comments
    script = server_document('http://localhost:5006/Grouped?file=' + fileLocation)
    return render_to_response('pages/visualization1.html', dict(script=script))


def weightedgraph(request):

    fileLocation = request.COOKIES['Data']
    # store comments
    script = server_document('http://localhost:5006/Weighted')
    return render_to_response('pages/visualization2.html', dict(script=script))


def faq(request):
    return render(request, 'pages/FAQ.html')


def data(request):
    return render(request, 'pages/data.html')


def loadData(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
    return render(request, 'pages/loadData.html')


def data_list(request):
    datasets = Data.objects.all()
    return render(request, 'pages/data_list.html', {
        'datasets': datasets
    })


def upload_data(request):
    if request.method == 'POST':
        form = DataForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('data_list')
    else:
        form = DataForm()
    return render(request, 'pages/upload_data.html', {
        'form': form
    })


def delete_data(request, pk):
    if request.method == 'POST':
        data = Data.objects.get(pk=pk)
        data.delete()
    return redirect('data_list')

def step1(request):
    return render(request, 'pages/step1.html')


def step2(request):
    return render(request, 'pages/step2.html')


def adjacencymatrix(request):

    # store comments

    script = server_document('http://localhost:5006/Adjacent')
    return render_to_response('pages/visualization3.html', dict(script=script))



