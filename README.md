"# WSP"
1) Установка БД
3) Установить rabbitmq
4) Создать venv и установить все пакеты
5) Ввести настройки в WSP/WSP/settings.py
Далее все исполняется из venv (для входа source venv/bin/activate)
6) Сделать миграции:
python manage.py makemigrations
python manage.py migrate
7) Запустить воркер:
celery -A WSP worker -l info -B
8) Запустить сервер:
python manage.py runserver


Установка БД {
sudo apt update
sudo apt install postgresql postgresql-contrib pgadmin3
sudo -i -u postgres
createdb wsp
psql
\password postgres
\q
}

Создание venv{
sudo apt install python3 python3-pip python3-venv
cd WSP
pyvenv venv
source venv/bin/activate
pip install req.txt
}

Создание администратора {
python manage.py createsuperuser
далее создавать можно через mysite/admin

}
