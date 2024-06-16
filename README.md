# Video search engine
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
# Использование
User-friendly: открыть страницу http://127.0.0.1:6001/
Machine-friendly:
`http://127.0.0.1:6001/searchVideo?k=10&query=котики` -- поиск среди проиндексированных видео
`http://127.0.0.1:6001/addVideoHandle?url=https....mp4&description=roblox` -- индексация видео
На Питоне:
```
import requests
requests.get("http://127.0.0.1:6001/addVideoHandle",params=[("url",your_url),("description","roblox")]) # Индексация
requests.get("http://127.0.0.1:6001/searchVideo",params=[("k",10),("query","котики")]) # Поиск
```
