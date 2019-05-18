from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
 # path("", views.homepage, name="homepage"),
 url(r'^$', views.homepage, name="home"),
 url(r'^visualization1.html/$', views.coauthorship, name="vis1"),
 url(r'^visualization2.html/$', views.weightedgraph, name='vis2'),
url(r'^visualization3.html/$', views.adjacencymatrix, name='vis3'),
 url(r'^data/$', views.data, name="data"),
 url(r'^FAQ/$', views.faq, name="faq"),
 url(r'^step1/$', views.step1, name="step1"),
 # path('visualization1.html/', views.coauthorship),
 # path('visualization2.html/', views.weightedgraph),
 # path('visualization3.html/', views.adjacencymatrix),
]
