# Create your tasks here
from __future__ import absolute_import, unicode_literals
from celery import shared_task
from .models import City, Illness, KerasModel, UntrainedModel, AggregatedDiseaseDaily, AggregatedDisease, Tasker
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import losses
from pandas import DataFrame
from uuid import uuid4
from datetime import datetime

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100


@shared_task
def trainer(model_id, namemodel, description, cityid, illnessid, weekly):
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

    if weekly:
        diseases = AggregatedDisease.objects.filter(city=city, illness=illness).order_by('date')
        counts = [x.count for x in diseases]
        dates = [x.date for x in diseases]
        name = {'Count': counts}
        dataframe = DataFrame.from_dict(name)
        dataframe = dataframe.rolling(window=8, on='Count').mean()
        dataframe = dataframe.pct_change()
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
        kmodel.save()
        task.timeEnd = datetime.now()
        task.result = 'Модель успешно обучена'
        task.save()

    return 1


@shared_task
def mul(x, y):
    return x * y


@shared_task
def xsum(numbers):
    return sum(numbers)