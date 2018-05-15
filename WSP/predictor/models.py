from django.contrib.postgres.fields import JSONField
from django.contrib.postgres.fields import ArrayField
from django.db import models

# Create your models here.


class City(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return '%s' % (self.name)

class Illness(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return '%s' % (self.name)

class UntrainedModel(models.Model):
    mod = JSONField()
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=500)
    lookback = models.IntegerField()

class KerasModel(models.Model):
    modelstructure = models.ForeignKey(UntrainedModel, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    city = models.ForeignKey(City, on_delete=models.CASCADE)
    illness = models.ForeignKey(Illness, on_delete=models.CASCADE)
    description = models.CharField(max_length=500)
    acc = models.IntegerField()
    hdfsig = models.CharField(max_length=200)
    traindate = models.DateTimeField()
    mindatadate = models.DateField(null=True)
    maxdatadate = models.DateField(null=True)
    weekly = models.BooleanField()



class Diseased(models.Model):
    sex = models.BooleanField()
    DOB = models.DateField()
    city = models.ForeignKey(City, on_delete=models.CASCADE)
    illness = models.ForeignKey(Illness, on_delete=models.CASCADE)
    date = models.DateTimeField()


class AggregatedDisease(models.Model):
    city = models.ForeignKey(City, on_delete=models.CASCADE)
    date = models.DateField()
    count = models.IntegerField()
    illness = models.ForeignKey(Illness, on_delete=models.CASCADE)


class AggregatedDiseaseDaily(models.Model):
    city = models.ForeignKey(City, on_delete=models.CASCADE)
    date = models.DateField()
    count = models.IntegerField()
    illness = models.ForeignKey(Illness, on_delete=models.CASCADE)


class Temperature(models.Model):
    date = models.DateTimeField()
    forecast = models.BooleanField()
    temp = models.IntegerField()
    city = models.ForeignKey(City, on_delete=models.CASCADE)


class DiseasePrediction(models.Model):
    city = models.ForeignKey(City, on_delete=models.CASCADE)
    date = models.DateField()
    count = models.IntegerField()
    illness = models.ForeignKey(Illness, on_delete=models.CASCADE)
    modelk = models.ForeignKey(KerasModel, on_delete=models.CASCADE)


class Tasker(models.Model):
    name = models.CharField(max_length=100)
    timeStart = models.DateTimeField()
    timeEnd = models.DateTimeField(null=True)
    result = models.CharField(max_length=50)
