# video search engine
Поисковый движок коротких видеоклипов (до 1 минуты) нацеленный на русскоязычную аудиторию

# Установка
1) Настоятельно рекомендуется завести виртуальное окружение:
  ```
  conda create -n videoSearch python==3.10 pip
  conda activate videoSearch
  ```
  Или
  ```
  python3 -m virtualenv .venv
  source .venv/bin/activate
  ```
2) Установка зависимостей:
   ```
   pip install -r requirements.txt
   ```
   Также необходимо скачать веса модели [GigaAM-RNNT](https://github.com/salute-developers/GigaAM):
   ```
   cd nn_utils
   bash downloader.sh
   ```
# Запуск
1) Запуск бекенд-сервера с тяжёлыми нейросетями:
   ```
   cd nn_utils
   python3 app.py
   ```
2) Запуск фронтенд-сервера непосредственно с поиском и индексацией видео:
   ```
   cd frontend
   python3 app.py
   ```
