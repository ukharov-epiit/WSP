from django.shortcuts import render, redirect, HttpResponse
from .forms import CityForm, DiseaseForm, CSVAggDisease, AddJSONmodel, TrainModel, CSVTemp, AddHDR5toModel
from .models import City, Illness, AggregatedDisease, AggregatedDiseaseDaily, UntrainedModel, KerasModel, Tasker, DiseasePrediction, Temperature, Keys
from .tasks import trainer, reader, predict
from pandas import read_csv
from datetime import datetime
from numpy import random
import os
import json
from uuid import uuid4
import numpy
from django.contrib.auth.decorators import login_required
# Create your views here.



@login_required
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

@login_required
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

@login_required
def handle_uploaded_file(f, des, typer='.csv'):
    with open(str(des)+typer, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@login_required
def newDiseaseAgragated(request):

    if request.method == "POST":
        form = CSVAggDisease(request.POST, request.FILES)
        if form.is_valid():
            kek = form.files['fileD']
            dest = random.randint(1, 100)
            handle_uploaded_file(kek, dest)
            dataframe = read_csv(str(dest) + '.csv', usecols=[0, 1, 2], engine='c')

            for d in dataframe.iterrows():
                date = datetime.strptime('%d' % (d[1]['epi_year'],) + '-' + '%d' % (d[1]['epi_week'],) + '-0', '%Y-%W-%w')
                count = d[1]['count']
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

@login_required
def LoadCityTemp(request):

    if request.method == "POST":
        form = CSVTemp(request.POST, request.FILES)
        if form.is_valid():
            kek = form.files['fileD']
            dest = random.randint(1, 100)
            handle_uploaded_file(kek, dest)
            dataframe = read_csv(str(dest) + '.csv', usecols=[0, 1, 5], engine='c')
            dataframe = dataframe.dropna()
            dataset = dataframe.as_matrix()
            for d in dataset:
                date = datetime.strptime(d[0], '%d.%m.%Y %H:%M')
                temp = 273.15 + int(d[1])
                city = City.objects.get(id=form.data['selectCity'])
                tryfind = Temperature.objects.filter(date=date, city=city)
                if len(tryfind) == 0:
                    agg = Temperature()
                    agg.temp = temp
                    agg.date = date
                    agg.city = city
                    agg.humidity = int(d[2])
                    agg.save()
                else:
                    tryfind[0].count = temp
                    tryfind[0].save()
            os.remove(str(dest) + '.csv')
            return render(request, 'predictor/newTemps.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/newTemps.html', {'form': form, 'success': 1})
    else:
        form = CSVTemp()

    return render(request, 'predictor/newTemps.html', {'form': form, 'success': 0})

@login_required
def newDiseaseAgragatedDaily(request):

    if request.method == "POST":
        form = CSVAggDisease(request.POST, request.FILES)
        if form.is_valid():
            kek = form.files['fileD']
            dest = random.randint(1, 100)
            handle_uploaded_file(kek, dest)
            dataframe = read_csv(str(dest) + '.csv', usecols=[0, 8], engine='c')
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

@login_required
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

@login_required
def TrainedH5(request):

    if request.method == "POST":
        form = AddHDR5toModel(request.POST, request.FILES)
        if form.is_valid():

            try:
                weather = form.data['weather']
            except:
                weather = False
            try:
                weekly = form.data['weekly']
            except:
                weekly = False

            kek = form.files['fileD']
            minT = form.files['pklT']
            minD = form.files['pklD']
            name = form.data['name']
            description = form.data['description']
            cityid = form.data['selectCity']
            illnessid = form.data['selectDisease']
            model_id = form.data['selectModel']
            guid = uuid4()
            guidmax = uuid4()
            guidmaxTt  = uuid4()
            handle_uploaded_file(kek, guid, '.h5')
            handle_uploaded_file(minT, guidmaxTt, '.pkl')
            handle_uploaded_file(minD, guidmax, '.pkl')
            kmodel = KerasModel()
            kmodel.city = cityid
            kmodel.illness = illnessid
            kmodel.name = name
            kmodel.description = description
            kmodel.hdfsig = str(guid) + ".h5"
            kmodel.traindate = datetime.now()
            kmodel.acc = 0
            kmodel.weekly = weekly
            kmodel.modelstructure = model_id
            kmodel.mindatadate = datetime.now()
            kmodel.maxdatadate = datetime.now()
            kmodel.minmax = str(guidmax) + '.pkl'
            kmodel.minmaxTemp = str(guidmaxTt) + '.pkl'
            kmodel.active = False
            kmodel.save()
            return render(request, 'predictor/trainedh5.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/trainedh5.html', {'form': form, 'success': 1})
    else:
        form = AddHDR5toModel()

    return render(request, 'predictor/trainedh5.html', {'form': form, 'success': 0})

@login_required
def listJSONModels(request):

    mods = UntrainedModel.objects.all().order_by('name')

    return render(request, 'predictor/untrainedModelList.html', {'mods': mods})

@login_required
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
            trainer.delay(model_id, name, description, cityid, illnessid, weekly, weather)
            #trainer(model_id, name, description, cityid, illnessid, weekly, weather)
            return render(request, 'predictor/trainmodel.html', {'form': form, 'success': 2})
        else:
            return render(request, 'predictor/trainmodel.html', {'form': form, 'success': 1})
    else:
        form = TrainModel()

    return render(request, 'predictor/trainmodel.html', {'form': form, 'success': 0})

@login_required
def listtrainedmodels(request):

    mods = KerasModel.objects.all().order_by('name')

    return render(request, 'predictor/trainedModelList.html', {'buffers': mods})

@login_required
def listcities(request):

    mods = City.objects.all().order_by('name')

    return render(request, 'predictor/citylist.html', {'buffers': mods})

@login_required
def listillness(request):

    mods = Illness.objects.all().order_by('name')

    return render(request, 'predictor/listillness.html', {'buffers': mods})

@login_required
def blank(request):

    return render(request, 'predictor/blank.html')

@login_required
def cityremove(request, cityid):

    City.objects.get(id=cityid).delete()

    return redirect(listcities)

@login_required
def illnessremove(request, illnessid):

    Illness.objects.get(id=illnessid).delete()

    return redirect(listillness)

@login_required
def untrainedmodelremove(request, modelid):

    UntrainedModel.objects.get(id=modelid).delete()

    return redirect(listJSONModels)

@login_required
def trainedmodelremove(request, modelid):

    model = KerasModel.objects.get(id=modelid)
    os.remove(model.hdfsig)
    os.remove(model.minmax)
    model.delete()

    return redirect(listtrainedmodels)

@login_required
def testtrainedmodel(request, modelid):

    #model = KerasModel.objects.get(id=modelid)
    task = reader.delay(modelid)
    #mae, mape, data = reader(modelid)
    mae,mape,data = task.get()

    return render(request, 'predictor/trainedview.html', {'data': data, 'mae': mae, 'mape': mape})

@login_required
def tasks(request):

    data = Tasker.objects.all().order_by('-timeStart')[:50]

    return render(request, 'predictor/tasks.html', {'blocks': data})

@login_required
def changeactivitymodel(request, modelid):

    data = KerasModel.objects.get(id=modelid)
    data.active = not data.active
    data.save()

    return redirect(listtrainedmodels)

@login_required
def predictactivity(request):

    predict.delay()

    return redirect(listtrainedmodels)

@login_required
def predictions(request):

    pred = DiseasePrediction.objects.all().order_by('-date')

    return render(request, 'predictor/predictions.html', {'blocks': pred})


def setCases(request):
    if request.method == 'POST':
        json_data = json.loads(request.body)

        key = json_data['key']

        try:
            go = Keys.objects.get(key=key)
        except Keys.DoesNotExist:
            go = None

        if go is not None:
            for case in json_data['case_counts']:
                date = datetime.strptime(case['date'], '%Y-%m-%d')
                count = case['num']
                city = City.objects.get(id=case['city'])
                illness = Illness.objects.get(id=case['illness'])
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
            resp = {
                'response': 'Успешно'
            }
        else:
            resp = {
                'response': 'Неверный ключ'
            }
        response = json.dump(resp)
        return HttpResponse(response, content_type='application/json')


def getPrediction(request):
    if request.method == 'POST':
        json_data = json.loads(request.body)

        key = json_data['key']

        try:
            go = Keys.objects.get(key=key)
        except Keys.DoesNotExist:
            go = None

        if go is not None:
            case = json_data
            city = City.objects.get(id=case['city'])
            illness = Illness.objects.get(id=case['illness'])
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
            resp = {
                'response': 'Успешно'
            }
        else:
            resp = {
                'response': 'Неверный ключ'
            }
        response = json.dump(resp)
        return HttpResponse(response, content_type='application/json')


