from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
 # path("", views.homepage, name="homepage"),
 url(r'^$', views.homepage, name="home"),
 url(r'^visualization1/$', views.coauthorship, name="vis1"),
 url(r'^visualization2/$', views.weightedgraph, name='vis2'),
 url(r'^data/$', views.data, name="data"),
 url(r'^FAQ/$', views.faq, name="faq"),
 # path('visualization1.html/', views.coauthorship),
 # path('visualization2.html/', views.weightedgraph),
]
