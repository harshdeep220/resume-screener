from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('run-pipeline/', views.run_pipeline, name='run_pipeline'),
]
