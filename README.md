"# WSP"
1) Установить postgresql
2) Создать там БД "wsp"
3) Установить rabbitmq
4) Поставить все пакеты командой "pip install -r requirements.txt"
5) Создать пользователя БД. Настройки пользователя в WSP/WSP/settings.py
6) Сделать миграции:
python manage.py makemigrations
python manage.py migrate
7) Запустить воркер:
celery -A WSP worker -l info -B
8) Запустить сервер:
python manage.py runserver
