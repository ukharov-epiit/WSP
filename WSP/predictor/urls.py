from django.urls import path

from . import views

urlpatterns = [
    path('', views.blank, name='blank'),
    path('newcity/', views.newcity, name='newcity'),
    path('newdisease/', views.newdisease, name='newdisease'),
    path('citylist/', views.listcities, name='citylist'),
    path('illnesslist/', views.listillness, name='illnesslist'),
    path('newdiseasedata/', views.newDiseaseAgragated, name='newdiseasedata'),
    path('newdiseasedataDaily/', views.newDiseaseAgragatedDaily, name='newdiseasedatadaily'),
    path('newjsonmodel/', views.newJSONModel, name='newjsonmodel'),
    path('listjsonmodel/', views.listJSONModels, name='listjsonmodel'),
    path('trainmodel/<int:model_id>/', views.trainModel, name='trainmodel'),
    path('trainedmodellist/', views.listtrainedmodels, name='trainedmodellist'),
    path('deletecity/<int:cityid>/', views.cityremove, name='deletecity'),
    path('deleteillness/<int:illnessid>/', views.illnessremove, name='deleteillness'),
    path('deletetrained/<int:modelid>/', views.trainedmodelremove, name='deletetrained'),
    path('deleteuntrained/<int:modelid>/', views.untrainedmodelremove, name='deleteuntrained'),
    path('trainedview/<int:modelid>/', views.testtrainedmodel, name='trainedview'),
    path('tasks/', views.tasks, name='tasks'),
    path('activationmodel/<int:modelid>/', views.changeactivitymodel, name='activationmodel'),
    path('predictall/', views.predictactivity, name='predictactive'),
    path('predictions/', views.predictions, name='predictions'),
    path('newtemps/', views.LoadCityTemp, name='newtemps'),
    path('loadmodelh5/', views.TrainedH5, name='loadmodelh5'),
]
