from django.shortcuts import render
from .forms import CityForm, DiseaseForm, CSVAggDisease, AddJSONmodel, TrainModel
from .models import City, Illness, AggregatedDisease, AggregatedDiseaseDaily, UntrainedModel
from .tasks import add
from pandas import read_csv
from datetime import datetime
from numpy import random
import json
import numpy
# Create your views here.


def newcity(request):
    kek = add.delay(2, 3)
    if request.method == "POST":
        form = CityForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            if len(City.objects.filter(name=post.name).all()) == 0:
                post.save()
            return render(request, 'predictor/newCity.html', {'form': form})

    else:
        form = CityForm()
    return render(request, 'predictor/newCity.html', {'form': form})


def newdisease(request):
    if request.method == "POST":
        form = DiseaseForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            if len(Illness.objects.filter(name=post.name).all()) == 0:
                post.save()
            return render(request, 'predictor/newCity.html', {'form': form})

    else:
        form = DiseaseForm()

    return render(request, 'predictor/newCity.html', {'form': form})


def handle_uploaded_file(f, des):
    with open(str(des)+'.csv', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def newDiseaseAgragated(request):

    if request.method == "POST":
        form = CSVAggDisease(request.POST, request.FILES)
        if form.is_valid():
            kek = form.files['fileD']
            dest = random.randint(1, 100)
            handle_uploaded_file(kek, dest)
            dataframe = read_csv(str(dest) + '.csv', usecols=[0, 1, 2], engine='python', skipfooter=1)
            dataset = dataframe.as_matrix()
            for d in dataset:
                date = datetime.strptime('%d' % (d[0],) + '-' + '%d' % (d[1],) + '-0', '%Y-%W-%w')
                count = d[2]
                city = City.objects.get(id=form.data['selectCity'])
                illness = Illness.objects.get(id=form.data['selectDisease'])
                tryfind = AggregatedDisease.objects.filter(date=date, city=city, illness=illness)
                if len(tryfind) == 0:
                    agg = AggregatedDisease()
                    agg.count = count
                    agg.date = date
                    agg.city = city
                    agg.illness = illness
                    agg.save()
                else:
                    tryfind[0].count = count
                    tryfind[0].save()

            return render(request, 'predictor/newFile.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/newFile.html', {'form': form, 'success': 1})
    else:
        form = CSVAggDisease()

    return render(request, 'predictor/newFile.html', {'form': form, 'success': 0})


def newDiseaseAgragatedDaily(request):

    if request.method == "POST":
        form = CSVAggDisease(request.POST, request.FILES)
        if form.is_valid():
            kek = form.files['fileD']
            dest = random.randint(1, 100)
            handle_uploaded_file(kek, dest)
            dataframe = read_csv(str(dest) + '.csv', usecols=[0, 8], engine='python', skipfooter=1)
            dataset = dataframe.as_matrix()
            for d in dataset:
                date = datetime.strptime(d[0], '%Y-%m-%d')
                count = d[1]
                city = City.objects.get(id=form.data['selectCity'])
                illness = Illness.objects.get(id=form.data['selectDisease'])
                tryfind = AggregatedDiseaseDaily.objects.filter(date=date, city=city, illness=illness)
                if len(tryfind) == 0:
                    agg = AggregatedDiseaseDaily()
                    agg.count = count
                    agg.date = date
                    agg.city = city
                    agg.illness = illness
                    agg.save()
                else:
                    tryfind[0].count = count
                    tryfind[0].save()

            return render(request, 'predictor/newFile.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/newFile.html', {'form': form, 'success': 1})
    else:
        form = CSVAggDisease()

    return render(request, 'predictor/newFile.html', {'form': form, 'success': 0})


def newJSONModel(request):

    if request.method == "POST":
        form = AddJSONmodel(request.POST, request.FILES)
        if form.is_valid():
            modelfile = form.files['fileD'].read()
            test = json.loads(modelfile)
            modelfile = json.dumps(test)
            name = form.data['name']
            description = form.data['description']
            untrained = UntrainedModel()
            untrained.name = name
            untrained.mod = modelfile
            untrained.description = description
            untrained.save()

            return render(request, 'predictor/newJSON.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/newJSON.html', {'form': form, 'success': 1})
    else:
        form = AddJSONmodel()

    return render(request, 'predictor/newJSON.html', {'form': form, 'success': 0})


def listJSONModels(request):

    mods = UntrainedModel.objects.all()

    return render(request, 'predictor/untrainedModelList.html', {'mods': mods})


def trainModel(request):

    if request.method == "POST":
        form = TrainModel(request.POST, request.FILES)
        if form.is_valid():


            return render(request, 'predictor/trainmodel.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/trainmodel.html', {'form': form, 'success': 1})
    else:
        form = TrainModel()

    return render(request, 'predictor/trainmodel.html', {'form': form, 'success': 0})
