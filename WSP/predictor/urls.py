from django.urls import path

from . import views


urlpatterns = [
    path('', views.blank, name='blank'),
    path('newcity/', views.newcity, name='newcity'),
    path('newdisease/', views.newdisease, name='newdisease'),
    path('newdiseasedata/', views.newDiseaseAgragated, name='newdiseasedata'),
    path('newdiseasedataDaily/', views.newDiseaseAgragatedDaily, name='newdiseasedatadaily'),
    path('newjsonmodel/', views.newJSONModel, name='newjsonmodel'),
    path('listjsonmodel/', views.listJSONModels, name='listjsonmodel'),
    path('trainmodel/<int:model_id>/', views.trainModel, name='trainmodel'),
    path('trainedmodellist/', views.listtrainedmodels, name='trainedmodellist'),
]