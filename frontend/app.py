#!/usr/bin/env python3
import sys,time,os,tempfile,requests,json
from flask import Flask
from flask import request
import numpy as np
import pickle

# Подгрузка ранее проиндексированных видео
# Здесь это дамп списка словарей, но решение дружелюбно к векторным базам данных и обычному поиску по тексту

dict_store = []
try:
	with open("videostore.pkl","rb") as fp:
		dict_store = pickle.load(fp)
except:
	print("Can't load anything. White paper! Index your videos")

all_urls = [o['url'] for o in dict_store]

# На моём CPU текстовый энкодер текста исполняется за 300 мс, кеширование позволяет сэкономить время
text2vec_cache = {}

def text2vec(text):
	'''
	Текстовый энкодер CLIP, обращение к API бекенда-сервера
	'''
	global text2vec_cache
	if text in text2vec_cache:
		return text2vec_cache[text]
	res = requests.get("http://127.0.0.1:6000/clip_text_encode",params=[("text",text)])
	res = json.loads(res.text)
	tmp = np.array(res["embedding"])
	text2vec_cache[text] = tmp
	return text2vec_cache[text]

doc2vec_cache = {}

def doc2vec(text):
	'''
	Текстовый энкодер doc2vec, обращение к API бекенда-сервера
	'''
	global doc2vec_cache
	if text in doc2vec_cache:
		return doc2vec_cache[text]
	res = requests.get("http://127.0.0.1:6000/doc2vec",params=[("text",text)])
	res = json.loads(res.text)
	res["embedding"] = np.array(res["embedding"])
	doc2vec_cache[text] = res
	return res

def video2dict(videopath,description="",stride=5,K=5):
	'''
	Обработка видео, обращение к API бекенда-сервера
	'''
	res = requests.get("http://127.0.0.1:6000/encodeVideoByLink",params=[
		("url",videopath),
		("description",description),
		("stride",stride),
		("K",K),
	])
	res = json.loads(res.text)
	if 'error' in res:
		print(res)
		return res
	res = res['result']
	for k in res.keys():
		if isinstance(res[k],list):
			res[k] = np.array(res[k])
	return res

import torch
import torch.nn as nn

# Ранжирование видео в выдаче зависит от запроса.
# Здесь предлагается простая модель выбора коэффициентов, с которыми брать каждый из параметров близости
# Предобученные нами модели лежат в папке models
# Pipeline: запрос -> вектор -> понижение размерности с помощью PCA -> наша модель -> коэффициенты, с которыми свернуть вектора выдачи -> отсортировать выдачу
# Замечане 1, что при масштабировании решения эту модель было бы здорово усложнить,
# а запросы к БД (FAISS и пр.) можно делать асинхронно, не дожидаясь результатов от этой модели

sorter_model_doc = nn.Sequential(
	nn.Linear(5,11),
	nn.Dropout(p=0.2),
	nn.Sigmoid() # -> (0,1) Важно! Модель не допускает отрицательных или запредельно больших коэффициентов, это сделает обучение невозможным
)

print("sorter_model:", sorter_model_doc.load_state_dict(torch.load('models/sorter_model_doc.pt')) )

pca_M_doc = np.loadtxt("models/pca_M_doc.dat")
pca_mu_doc = np.loadtxt("models/pca_mu_doc.dat")

def getCoefsByPrompt_doc(query):
	'''
	Преобразует текстовый запрос в коэффициенты от модели
	Вход: текст или вектор текстового эмбеддинга
	Выход: NumPy массив из 11 чисел
	'''
	if type(query)==str:
		query = doc2vec(query)['embedding']
	sorter_model_doc.eval()
	enc = (query.reshape(1,-1)-pca_mu_doc).dot(pca_M_doc)
	with torch.no_grad():
		enc = torch.Tensor(enc)
		coefs = sorter_model_doc(enc)
		return coefs.numpy()[0]

# Аналогично для текстового энкодера CLIPа похожая модель:
pca_M_clip = np.loadtxt("models/pca_M_clip.dat")
pca_mu_clip = np.loadtxt("models/pca_mu_clip.dat")

sorter_model_clip = nn.Sequential(
	nn.Linear(6,11),
	nn.Dropout(p=0.2),
	nn.Sigmoid() # -> (0,1) important!
)

print("sorter_model:",sorter_model_clip.load_state_dict(torch.load('models/sorter_model_clip.pt')) )

def getCoefsByPrompt_clip(query):
	if type(query)==str:
		query = text2vec(query)
	sorter_model_clip.eval()
	enc = (query.reshape(1,-1)-pca_mu_clip).dot(pca_M_clip)
	with torch.no_grad():
		enc = torch.Tensor(enc)
		coefs = sorter_model_clip(enc)
		return coefs.numpy()[0]

app = Flask(__name__)

@app.route("/")
def hello():
	return open("index.html","r").read()

@app.route("/addVideoHandle", methods=['GET','POST'])
def addVideo():
	start_t = time.time()
	url = request.args.get('url')
	description = request.args.get('description',"")
	stride = int(request.args.get('stride',5)) # Опционально
	K = int(request.args.get('K',5)) # Опционально
	force = bool(request.args.get('force',False)) # Принудительно поторно индексирует видео, если оно уже есть в индексе
	if url == "": # Если в поле URL пусто, пытаемся извлечь данные из поля данных POST-запроса
		post_data = request.get_json()
		url = post_data['url']
		description = post_data['description']
	res = dict()
	if not force and url in all_urls:
		res['result'] = 'fail'
		res['error'] = 'Video with this url already indexed'
		return res
	try:
		tmp_obj = video2dict(url,description=description,stride=stride,K=K)
		if 'error' in tmp_obj:
			raise Exception(tmp_obj['error'])
		tmp_obj['url'] = url
		tmp_obj['desc_orig'] = description
		dict_store.append(tmp_obj) # После отработки этого запроса видео уже можно искать в поиске
		all_urls.append(url)
		res['result'] = 'ok'
		res['storelen'] = len(dict_store)
		with open("videostore.pkl","wb") as fp:
			pickle.dump(dict_store, fp)
	except Exception as e:
		res['result'] = 'fail'
		res['error'] = str(e)
	res['time'] = time.time()-start_t
	return res

def preprocessVideoForQuery(iVideo,clip_emb,doc_emb,doc_lemmed,doc_words):
	'''
	Извлечение степени схожести видео на текстовый запрос.
	При масштабировании перенести эту функцию на движок БД
	Входы:
	iVideo -- словарь со свойствами видео
	clip_emb -- текстовый эмбеддинг от CLIP
	doc_emb -- текстовый эмбеддинг от doc2vec
	doc_lemmed -- множество лемматизированных слов
	doc_words -- слова запроса без лемматизации
	Выход: список из 11 чисел не более 1 каждое
	'''
	features = []
	clip_innerprod = iVideo['clip_embeddings'].dot(clip_emb)
	features.append(clip_innerprod.min())
	features.append(clip_innerprod.mean())
	features.append(clip_innerprod.max())
	features.append(clip_innerprod.std())
	doc_innerprod = iVideo['description_embedding'].dot(doc_emb)
	features.append(doc_innerprod)
	doc_innerprod2 = iVideo['transcription_embedding'].dot(doc_emb)
	features.append(doc_innerprod2)
	features.append(max(features))
	# 2.2) Words calculating
	description_words = set(iVideo['description'].lower().split(" "))
	description_words_comp = len(description_words & doc_words) / max(len(doc_words),1)
	features.append(description_words_comp)
	transcription_words = set(iVideo['transcription'].split(" "))
	transcription_words_comp = len(transcription_words & doc_words) / max(len(doc_words),1)
	features.append(transcription_words_comp)
	transcription_words2 = set(iVideo['transcription_lemmed'])
	transcription_words_comp2 = len(transcription_words2 & doc_lemmed) / max(len(doc_lemmed),1)
	features.append(transcription_words_comp2)
	features.append(max(features))
	return features

import re
url_parse = re.compile("https\\:\\/\\/\\S+fhd\.mp4")

def combo_search(query,k=10):
	'''
	Поиск топ k видео по текстовому запросу query
	'''
	start_t = time.time()
	clip_emb = text2vec(query.replace("#",""))
	doc_enc = doc2vec(query)
	doc_emb = doc_enc['embedding']
	doc_lemmed = set(doc_enc['lemmed'] if 'lemmed' in doc_enc else [])
	doc_words = set(query.replace(",","").lower().split(" "))
	urls = url_parse.findall(query)
	if len(urls)>0: # Фича поиска похожих видео, протестировано слабо, но результатом я доволен
		urls_len = sum([len(u) for u in urls])
		clip_emb *= (len(query)-urls_len)/len(query)
		doc_emb *= (len(query)-urls_len)/len(query)
		for iVideo in dict_store:
			if iVideo['url'] in urls:
				clip_emb+=iVideo['clip_embeddings'].mean(axis=0)*urls_len/len(query)
				doc_emb+=iVideo['description_embedding']*urls_len/len(query)
	preprocessed_videos_codes = []
	distances_list = []
	hashtags = [w for w in doc_words if w.startswith('#')]
	weights = 0.6*getCoefsByPrompt_doc(doc_emb) # Получение коэффициентов для ранжирования
	weights += 0.4*getCoefsByPrompt_clip(clip_emb)
	encoding_time = time.time()-start_t
	search_time = time.time()
	for iVideo in dict_store:
		preprocessed_videos_codes.append( preprocessVideoForQuery(iVideo,clip_emb,doc_emb,doc_lemmed,doc_words) )
		cur_hashs = iVideo['description'].lower().replace(","," ")+" "
		hash_sum = sum([int(w in cur_hashs) for w in hashtags])
		if iVideo['description']=='nan':
			iVideo['description']=''
		distances_list.append({'url':iVideo['url'],
							'description':iVideo['description'],
							'hashtags':hash_sum/max(len(cur_hashs),len(hashtags),1),
							'metrics':preprocessed_videos_codes[-1]
							})
	preprocessed_videos_codes = np.array(preprocessed_videos_codes)
	weighted_sum = preprocessed_videos_codes.dot(weights)
	for i in range(len(distances_list)):
		distances_list[i]['weighted'] = weighted_sum[i]
	distances_list.sort(key = lambda x:(-x['hashtags'],-x['weighted']) ) # Сначала упорядочить по хештегам дословно
	search_time = time.time()-search_time
	return distances_list[:k],weights.tolist(),encoding_time,search_time

@app.route("/searchVideo", methods=['GET','POST'])
def searchVideoHandle():
	start_t = time.time()
	prompt = request.args.get('query')
	k = int(request.args.get('k',10))
	try:
		learn_data = request.get_data(as_text=True)
		print("data:",learn_data)
		if learn_data is not None and len(learn_data)>5: # Записать голоса релевантноси в файл для дообучения моделей
			learn_data = learn_data.split(",")
			with open("search_alignment.csv","a") as fl:
				for l in learn_data:
					print(l.replace("|>",";"),file=fl)
	except:
		print("No data passed")
	res = dict()
	try:
		tmp_obj,prompt_coeffs,encoding_time,search_time = combo_search(prompt,k=k)
		res['result'] = tmp_obj
		res['prompt_coeffs'] = prompt_coeffs
	except Exception as e:
		print(e)
		res['result'] = {}
		res['error'] = str(e)
	res['all_videos'] = len(dict_store)
	res['time'] = [round((time.time()-start_t)*1000,2),round(encoding_time*1000,2),round(search_time*1000,2)]
	return res

if __name__ == "__main__":
	app.run(host='127.0.0.1',debug=False,port=6001,use_reloader=False)
