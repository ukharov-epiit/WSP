from django import forms
from .models import City, Illness


class CityForm(forms.ModelForm):

    class Meta:
        model = City
        fields = ('name',)
        labels = {
            "name": "Название города"
        }


class DiseaseForm(forms.ModelForm):

    class Meta:
        model = Illness
        fields = ('name',)
        labels = {
            "name": "Название болезни"
        }

def getIllCityChoices():
    choiceIll = []
    for choice in Illness.objects.all():
        choiceIll.append((choice.id, choice.name))

    choiceCity = []
    for choice in (City.objects.all()):
        choiceCity.append((choice.id, choice.name))

    return choiceIll, choiceCity


class CSVAggDisease(forms.Form):

    choiceIll, choiceCity = getIllCityChoices()

    selectDisease = forms.CharField(widget=forms.Select(choices=choiceIll), label='Заболевание')
    selectCity = forms.CharField(widget=forms.Select(choices=choiceCity), label='Город')

    fileD = forms.FileField(label='Данные')


class AddJSONmodel(forms.Form):

    name = forms.CharField(label='Название')
    description = forms.CharField(widget=forms.Textarea(), label='Описание')
    fileD = forms.FileField(label='Модель JSON')


class TrainModel(forms.Form):
    choiceIll, choiceCity = getIllCityChoices()
    name = forms.CharField(label='Название')
    description = forms.CharField(widget=forms.Textarea(), label='Описание')
    selectDisease = forms.CharField(widget=forms.Select(choices=choiceIll), label='Заболевание')
    selectCity = forms.CharField(widget=forms.Select(choices=choiceCity), label='Город')
    weekly = forms.BooleanField(label='Недельные данные')
    weather = forms.BooleanField(label='Включить погоду')
