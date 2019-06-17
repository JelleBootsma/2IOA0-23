from django.urls import path
from django.conf.urls import url
from . import views
from django.contrib import admin
from application import views
from sampledata_vis import settings
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
                # path("", views.homepage, name="homepage"),
                url(r'^$', views.homepage, name="home"),
                url(r'^visualization1.html/$', views.coauthorship, name="vis1"),
                url(r'^visualization2.html/$', views.weightedgraph, name='vis2'),
                url(r'^visualization3.html/$', views.adjacencymatrix, name='vis3'),
                url(r'^visualization4.html/$', views.hierarchical, name='vis4'),
                url(r'^data.html/$', views.data, name="data"),
                url(r'^aboutus.html/$', views.aboutus, name="aboutus"),
                url(r'^step1/$', views.step1, name="step1"),
                url(r'^step2/$', views.step2, name="step2"),
                url(r'^loadData.html/', views.loadData, name="loadData"),
                url(r'^data_list/', views.data_list, name='data_list'),
                url(r'^upload_data/', views.upload_data, name='upload_data'),
                path("data/<int:pk>/", views.delete_data, name='delete_data')
              ] #+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
               # path('visualization1.html/', views.coauthorship),
               # path('visualization2.html/', views.weightedgraph),
               # path('visualization3.html/', views.adjacencymatrix),

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


