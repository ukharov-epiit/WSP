from django.shortcuts import render, redirect
from .forms import CityForm, DiseaseForm, CSVAggDisease, AddJSONmodel, TrainModel
from .models import City, Illness, AggregatedDisease, AggregatedDiseaseDaily, UntrainedModel, KerasModel, Tasker, DiseasePrediction
from .tasks import trainer, reader, predict
from pandas import read_csv
from datetime import datetime
from numpy import random
import os
import json
import numpy
# Create your views here.




def newcity(request):
    if request.method == "POST":
        form = CityForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            if len(City.objects.filter(name=post.name).all()) == 0:
                post.save()
            return render(request, 'predictor/newCity.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/newCity.html', {'form': form, 'success': 1})
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
            return render(request, 'predictor/newCity.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/newCity.html', {'form': form, 'success': 1})
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
            os.remove(str(dest) + '.csv')
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
            os.remove(str(dest) + '.csv')
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
            lookback = form.data['lookback']
            untrained = UntrainedModel()
            untrained.name = name
            untrained.mod = modelfile
            untrained.description = description
            untrained.lookback = lookback
            untrained.save()

            return render(request, 'predictor/newJSON.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/newJSON.html', {'form': form, 'success': 1})
    else:
        form = AddJSONmodel()

    return render(request, 'predictor/newJSON.html', {'form': form, 'success': 0})


def listJSONModels(request):

    mods = UntrainedModel.objects.all().order_by('name')

    return render(request, 'predictor/untrainedModelList.html', {'mods': mods})


def trainModel(request, model_id):

    if request.method == "POST":
        form = TrainModel(request.POST)
        if form.is_valid():
            name = form.data['name']
            description = form.data['description']
            cityid = form.data['selectCity']
            illnessid = form.data['selectDisease']
            try:
                weather = form.data['weather']
            except:
                weather = False
            try:
                weekly = form.data['weekly']
            except:
                weekly = False
            trainer.delay(model_id, name, description, cityid, illnessid, weekly)
            return render(request, 'predictor/trainmodel.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/trainmodel.html', {'form': form, 'success': 1})
    else:
        form = TrainModel()

    return render(request, 'predictor/trainmodel.html', {'form': form, 'success': 0})


def listtrainedmodels(request):

    mods = KerasModel.objects.all().order_by('name')

    return render(request, 'predictor/trainedModelList.html', {'buffers': mods})


def listcities(request):

    mods = City.objects.all().order_by('name')

    return render(request, 'predictor/citylist.html', {'buffers': mods})


def listillness(request):

    mods = Illness.objects.all().order_by('name')

    return render(request, 'predictor/listillness.html', {'buffers': mods})

def blank(request):

    return render(request, 'predictor/blank.html')

def cityremove(request, cityid):

    City.objects.get(id=cityid).delete()

    return redirect(listcities)


def illnessremove(request, illnessid):

    Illness.objects.get(id=illnessid).delete()

    return redirect(listillness)


def untrainedmodelremove(request, modelid):

    UntrainedModel.objects.get(id=modelid).delete()

    return redirect(listJSONModels)


def trainedmodelremove(request, modelid):

    model = KerasModel.objects.get(id=modelid)
    os.remove(model.hdfsig)
    os.remove(model.minmax)
    model.delete()

    return redirect(listtrainedmodels)


def testtrainedmodel(request, modelid):

    model = KerasModel.objects.get(id=modelid)
    task = reader.delay(modelid)

    mae,mape,data = task.get()

    return render(request, 'predictor/trainedview.html', {'data': data, 'mae': mae, 'mape': mape})


def tasks(request):

    data = Tasker.objects.all().order_by('-timeStart')[:50]

    return render(request, 'predictor/tasks.html', {'blocks': data})


def changeactivitymodel(request, modelid):

    data = KerasModel.objects.get(id=modelid)
    data.active = not data.active
    data.save()

    return redirect(listtrainedmodels)


def predictactivity(request):

    predict.delay()

    return redirect(listtrainedmodels)


def predictions(request):

    pred = DiseasePrediction.objects.all().order_by('-date')

    return render(request, 'predictor/predictions.html', {'blocks': pred})