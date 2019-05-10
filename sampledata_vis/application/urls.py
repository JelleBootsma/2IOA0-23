from django.urls import path
from . import views

urlpatterns = [
 path("", views.homepage, name="homepage"),
 path('visualization1.html/', views.coauthorship),
 path('visualization2.html/', views.weightedgraph),
]
