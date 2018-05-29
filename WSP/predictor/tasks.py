# Create your tasks here
from __future__ import absolute_import, unicode_literals
from celery import shared_task
from .models import City, Illness, KerasModel, UntrainedModel, AggregatedDiseaseDaily, AggregatedDisease, Tasker, DiseasePrediction, Temperature, TemperaturePredicted
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.externals import joblib
from keras import losses
from pandas import DataFrame
from uuid import uuid4
from datetime import datetime, timedelta
import json as jason
import requests
import time
from celery import Task
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def create_datasetAware(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back,0])
    return numpy.array(dataX), numpy.array(dataY)

def create_predataset(dataset, look_back=3):
    dataX  = []
    ll = len(dataset)
    for i in range(len(dataset) - look_back):
        a = dataset[i+1:(i+1 + look_back), 0]
        dataX.append(a)
    return numpy.array(dataX)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100




@shared_task
def trainer(model_id, namemodel, description, cityid, illnessid, weekly, weather):
    # fix random seed for reproducibility

    numpy.random.seed(7)
    city = City.objects.get(id=cityid)
    illness = Illness.objects.get(id=illnessid)
    modelk = UntrainedModel.objects.get(id=model_id)
    task = Tasker()
    task.name = 'Тренировка модели ' + modelk.name
    task.timeStart = datetime.now()
    task.result = 'Задача поставлена'
    task.save()

    if weekly == 'on':
        weekly = True
    else:
        weekly = False

    if weather == 'on':
        weather = True
    else:
        weather = False

    if weekly and  not weather:
        diseases = AggregatedDisease.objects.filter(city=city, illness=illness).order_by('date')
        counts = [x.count for x in diseases]
        dates = [x.date for x in diseases]
        name = {'Count': counts}
        dataframe = DataFrame.from_dict(name)
        dataframe = dataframe.rolling(window=4).mean()
        dataframe = dataframe.diff()
        dataframe = dataframe.dropna()
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        train_size = int(len(dataset) * 0.80)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        # reshape into X=t and Y=t+3
        look_back = modelk.lookback
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        activemodel = model_from_json(modelk.mod)
        activemodel.compile(loss=losses.mse, optimizer="adam")
        activemodel.fit(trainX, trainY, epochs=80, verbose=2)
        # make predictions
        trainPredict = activemodel.predict(trainX)
        testPredict = activemodel.predict(testX)

        # shift train predictions for plotting
        trainPredictPlot = numpy.empty_like(dataset)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(dataset)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

        testScore = mean_absolute_error(testY, testPredict[:, 0])
        print('Test Score: %.2f MAE' % (testScore))
        testScore_ = mean_absolute_percentage_error(testY, testPredict[:, 0])
        print('Test Score: %.2f MAPE' % (testScore_))
        # serialize weights to HDF5
        guid = uuid4()
        activemodel.save_weights(str(guid) + ".h5")
        guidmimax = uuid4()
        joblib.dump(scaler, str(guidmimax) +'.pkl')
        kmodel = KerasModel()
        kmodel.city = city
        kmodel.illness = illness
        kmodel.name = namemodel
        kmodel.description = description
        kmodel.hdfsig = str(guid) + ".h5"
        kmodel.traindate = datetime.now()
        kmodel.acc = 100 - testScore_
        kmodel.weekly = weekly
        kmodel.modelstructure = modelk
        kmodel.mindatadate = dates[0]
        kmodel.maxdatadate = dates[-1]
        kmodel.minmax = str(guidmimax) +'.pkl'
        kmodel.active = False
        kmodel.weather = False
        kmodel.save()
        task.timeEnd = datetime.now()
        task.result = 'Модель успешно обучена'

        task.save()
    elif not weather and not weekly:
        diseases = AggregatedDisease.objects.filter(city=city, illness=illness).order_by('date')
        counts = [x.count for x in diseases]
        dates = [x.date for x in diseases]
        name = {'Count': counts, 'Dates': dates}
        temps = Temperature.objects.filter(city=city, illness=illness).order_by('date')

        return 2
    return 1


@shared_task
def reader(model_id):
    # fix random seed for reproducibility

    numpy.random.seed(7)

    modelk = KerasModel.objects.get(id=model_id)
    json = modelk.modelstructure.mod

    diseases = AggregatedDisease.objects.filter(city=modelk.city, illness=modelk.illness).order_by('date')
    counts = [x.count for x in diseases]
    dates = [str(x.date) for x in diseases]
    name = {'Count': counts}
    dataframe = DataFrame.from_dict(name)
    dataframe = dataframe.rolling(window=4).mean()
    dataframe = dataframe.diff()
    dataframe = dataframe.dropna()
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    scaler = joblib.load(modelk.minmax)
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.80)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+3
    look_back = modelk.modelstructure.lookback
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    activemodel = model_from_json(modelk.modelstructure.mod)
    activemodel.load_weights(modelk.hdfsig)
    # make predictions
    trainPredict = activemodel.predict(trainX)
    testPredict = activemodel.predict(testX)

    testScore = mean_absolute_error(testY, testPredict[:, 0])
    print('Test Score: %.2f MAE' % (testScore))
    testScoreS = mean_squared_error(testY, testPredict[:, 0])
    print('Test Score: %.2f MSE' % (testScoreS))
    testScore_ = mean_absolute_percentage_error(testY, testPredict[:, 0])
    print('Test Score: %.2f MAPE' % (testScore_))

    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    trainY = scaler.inverse_transform([trainY])
    dataset = scaler.inverse_transform(dataset)

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    trainPredictPlot = numpy.nan_to_num(trainPredictPlot)
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    testPredictPlot = numpy.nan_to_num(testPredictPlot)

    data = {
        'labels': dates,
        'datasets': [{
            'data': testPredictPlot[:, 0].tolist(),
            'label': 'Тестовое прогнозирование',
            'fill': False,
            'backgroundColor': '#e29e17',
            'borderColor': '#e29e17'
        },
            {
                'data': trainPredictPlot[:, 0].tolist(),
                'label': 'Тренировочное прогнозирование',
                'fill': False,
                'backgroundColor': '#11dd4e',
                'borderColor': '#11dd4e'
            },
            {
                'data': dataset[:, 0].tolist(),
                'label': 'Актуальные данные',
                'fill': False,
                'backgroundColor': '#187ae2',
                'borderColor': '#187ae2'
            }]
    }
    jsondata = jason.dumps(data)

    return testScore, testScore_, jsondata

@shared_task
def predict():
    # fix random seed for reproducibility

    numpy.random.seed(7)

    models = KerasModel.objects.filter(active=True)
    for modelk in models:

        diseases = AggregatedDisease.objects.filter(city=modelk.city, illness=modelk.illness).order_by('-date')[:3*modelk.modelstructure.lookback+5]
        counts = list(reversed([x.count for x in diseases]))
        dates = list(reversed([x.date for x in diseases]))
        name = {'Count': counts}
        dataframe = DataFrame.from_dict(name)
        dataframe = dataframe.rolling(window=4).mean()
        dataframe = dataframe.diff()
        dataframe = dataframe.dropna()
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        scaler = joblib.load(modelk.minmax) #MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        # reshape into X=t and Y=t+3
        look_back = modelk.modelstructure.lookback
        trainX = create_predataset(dataset, look_back)
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        activemodel = model_from_json(modelk.modelstructure.mod)
        activemodel.load_weights(modelk.hdfsig)
        # make predictions
        trainPredict = activemodel.predict(trainX)
        trainPredict = scaler.inverse_transform(trainPredict)
        prediction = trainPredict[:, 0].tolist()[-1]
        disease = DiseasePrediction()
        disease.modelk = modelk
        disease.weekly = modelk.weekly
        disease.city = modelk.city
        disease.illness = modelk.illness
        disease.count = prediction*100
        if modelk.weekly:
            disease.date = dates[-1] + timedelta(days=7)
        else:
            disease.date = dates[-1] + timedelta(days=1)
        disease.save()

    return True


@shared_task
def load_temperature():
    key = 'd6bbc51396ecf08d4b745f198cb99c87'
    cities = City.objects.all()
    for city in cities:
        request = requests.get('http://api.openweathermap.org/data/2.5/weather?q=' + city.internationalName +'&appid=' + key)
        if request.status_code == 200:
            data = request.json()
            if data['cod'] == '200':
                temp = Temperature()
                temp.city = city
                temp.date = datetime.now()
                temp.temp = data['main']['temp']
                temp.humidity = data['main']['humidity']
                temp.save()
        time.sleep(1)

    return 'success'


@shared_task
def load_forecast():
    key = 'd6bbc51396ecf08d4b745f198cb99c87'
    cities = City.objects.all()
    for city in cities:
        request = requests.get('http://api.openweathermap.org/data/2.5/forecast?q=' + city.internationalName +'&appid=' + key)
        if request.status_code == 200:
            data = request.json()
            if data['cod'] == '200':
                for i in data['list']:
                    temp = TemperaturePredicted()
                    temp.city = city
                    temp.date = datetime.strptime(i['dt_txt'], '%Y-%m-%d %H:%M:%S')
                    temp.temp = i['main']['temp']
                    temp.humidity = i['main']['humidity']
                    temp.save()
        time.sleep(1)
    return 'success'


@shared_task
def tetstask():
    print('lalala')