from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
import django
import requests
from datetime import datetime
import time
from celery.schedules import timedelta
# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'WSP.settings')

app = Celery('WSP')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

#django.setup()
#from predictor.tasks import load_forecast, load_temperature

delta = timedelta(hours=2)
deltaForecast = timedelta(hours=3)

app.conf.beat_schedule = {
    'load-forecast': {
        'task': 'predictor.tasks.load_forecast',
        'schedule': deltaForecast
    },
    'load-temp': {
        'task': 'predictor.tasks.load_temperature',
        'schedule': delta
    },
}



@app.task(bind=True)
def debug_task(self):
    print('Request: {0!r}'.format(self.request))


#@app.on_after_configure.connect
#def setup_periodic_tasks(sender, **kwargs):
#    # Calls load temps function every hour
#    sender.add_periodic_task(delta, load_temperature)
#    sender.add_periodic_task(deltaForecast, load_forecast)


@app.task
def test():
    print(1)


