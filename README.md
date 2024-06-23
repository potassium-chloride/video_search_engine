# Video search engine
Поисковый движок коротких видеоклипов (до 1 минуты) нацеленный на русскоязычную аудиторию

# Установка
1) Склонируйте репозиторий, перейдите в папку и запустите скрипт настройки install.sh. После этого активируйте виртуальное окружение:
  ```
  git clone https://github.com/potassium-chloride/video_search_engine/
  cd ./video_search_engine/
  bash ./install.sh
  source .venv/bin/activate
  ```
2) Перед первым запуском необходимо скачать веса модели [GigaAM-RNNT](https://github.com/salute-developers/GigaAM):
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
   При первом запуске скрипт скачает также веса CLIP и doc2vec моделей, это может занять некоторое время
2) Запуск фронтенд-сервера непосредственно с поиском и индексацией видео:
   ```
   cd frontend
   python3 app.py
   ```
# Использование
User-friendly: открыть страницу <http://127.0.0.1:6001/>.
На данный момент решение развёрнуто по адресу <http://85.113.39.151/yappy_search/>.
## Использование API:
### Поиск ранее загруженных видео:
Python:
```
import requests,json

query = "котики" # string, your search request
k = 10 # int, number of results

result = requests.get("http://85.113.39.151/yappy_search/searchVideo",
                   params=[
                                ("query",query),
                                ("k",k)
                          ])
print(result.status_code)
# 200 -- OK
# 504 -- Endpoint is down, ask me in direct messages
# 500 -- Bad request, check your parameters
result = result.json()['result']
for video in result:
    description = video['description']
    url = video['url']
    print(url,description) # print link and video description
```
Аналогичный запрос curl:
```
curl "http://85.113.39.151/yappy_search/searchVideo?k=10&query=котики"
```
### Индексация нового видео:
Python:
```
import requests,json

your_url = "https://cdn-st.rutubelist.ru/media/f4/8d/0766c7c04bb1abb8bf57a83fa4e8/fhd.mp4" # string, url for mp4 video
your_description = "#технологии #девайсы #technologies #гаджеты #смартчасы #умныечасы #миф" # string, description (optional)

result = requests.post("http://85.113.39.151/yappy_search/addVideoHandle",
                   params=[
                                ("url",your_url),
                                ("description",your_description)
                          ]) # It tooks about 25 seconds depends on video duration
print(result.status_code)
# 200 -- OK, you can parse JSON
# 504 -- Endpoint is down, ask me in direct messages
# 500 -- Bad request, check your parameters
result = result.json()
print(result['result']) # ok or fail
if 'error' in result:
    print("Error:",result['error'])
    # It could be raised by connection error and etc...
    # Note: it raises error if url already exist in db. Use parameter force=1 for indexing
else:
    print("Indexing time:",result['time']) # Time (seconds)
    print("Size of DB:",result['storelen']) # Count of all videos which has been indexed
```
