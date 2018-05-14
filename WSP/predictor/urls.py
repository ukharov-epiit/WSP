from django.urls import path

from . import views


urlpatterns = [
    path('newdity/', views.newcity, name='newcity'),
    path('newdisease/', views.newdisease, name='newdisease'),
    path('newdiseasedata/', views.newDiseaseAgragated, name='newdiseasedata'),
    path('newdiseasedataDaily/', views.newDiseaseAgragatedDaily, name='newdiseasedatadaily'),
    path('newjsonmodel/', views.newJSONModel, name='newjsonmodel'),
    path('listjsonmodel/', views.listJSONModels, name='listjsonmodel'),
    path('trainmodel/', views.trainModel, name='trainmodel'),
]