from django import forms
from .models import City, Illness, UntrainedModel, Keys


class CityForm(forms.ModelForm):

    class Meta:
        model = City
        fields = ('name', 'internationalName', 'latitude', 'longtitude' )
        labels = {
            "name": "Название города",
            "internationalName": "Латинское название",
            'latitude': 'Широта',
            'longtitude': 'Долгота'
        }

def getIllCityChoices():
    choiceIll = []
    choiceCity = []

    for choice in Illness.objects.all():
        choiceIll.append((choice.id, choice.name))
    for choice in (City.objects.all()):
        choiceCity.append((choice.id, choice.name))

    return choiceIll, choiceCity



class DiseaseForm(forms.ModelForm):

    class Meta:
        model = Illness
        fields = ('name',)
        labels = {
            "name": "Название болезни"
        }



class CSVAggDisease(forms.Form):

    selectDisease = forms.ModelChoiceField(queryset=Illness.objects.all().order_by('name'), label='Заболевание', widget=forms.Select(attrs={'class': 'form-control'}))
    selectCity = forms.ModelChoiceField(queryset=City.objects.all().order_by('name'), label='Город', widget=forms.Select(attrs={'class': 'form-control'}))

    fileD = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control-file'}), label='Данные')


class CSVTemp(forms.Form):

    selectCity = forms.ModelChoiceField(queryset=City.objects.all().order_by('name'), label='Город', widget=forms.Select(attrs={'class': 'form-control'}))

    fileD = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control-file'}), label='Данные')


class AddJSONmodel(forms.Form):

    name = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}), label='Название')
    description = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control'}), label='Описание')
    lookback = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}), label='Lookback')
    fileD = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control-file'}), label='Модель JSON')


class TrainModel(forms.Form):
    name = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}), label='Название')
    description = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control'}), label='Описание')
    selectDisease = forms.ModelChoiceField(queryset=Illness.objects.all().order_by('name'), label='Заболевание', widget=forms.Select(attrs={'class': 'form-control'}))
    selectCity = forms.ModelChoiceField(queryset=City.objects.all().order_by('name'), label='Город', widget=forms.Select(attrs={'class': 'form-control'}))
    weekly = forms.BooleanField(label='Недельные данные', initial='on', required=False)
    weather = forms.BooleanField(label='Включить погоду', initial='on', required=False)


class AddHDR5toModel(forms.Form):
    name = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}), label='Название')
    description = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control'}), label='Описание')
    selectModel = forms.ModelChoiceField(queryset=UntrainedModel.objects.all().order_by('name'), label='Модель',
                                           widget=forms.Select(attrs={'class': 'form-control'}))
    selectDisease = forms.ModelChoiceField(queryset=Illness.objects.all().order_by('name'), label='Заболевание', widget=forms.Select(attrs={'class': 'form-control'}))
    selectCity = forms.ModelChoiceField(queryset=City.objects.all().order_by('name'), label='Город', widget=forms.Select(attrs={'class': 'form-control'}))
    weekly = forms.BooleanField(label='Недельные данные', initial='on', required=False)
    weather = forms.BooleanField(label='Включить погоду', initial='on', required=False)
    fileD = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control-file'}), label='Веса H5')
    pklD = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control-file'}), label='Скейлер данных')
    pklT = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control-file'}), label='Скейлер погоды', required=False)