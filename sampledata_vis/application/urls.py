from django.urls import path
from . import views

urlpatterns = [
 path("", views.homepage, name="homepage"),
 path('visualization1.html/', views.visualization1),
 path('visualization2.html/', views.visualization2),
]
