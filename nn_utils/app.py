#!/usr/bin/env python3
import sys,time,os,tempfile,requests
from flask import Flask
from flask import request

import torch

app = Flask(__name__)

@app.route("/")
def hello():
	return "Hello, World!"

############# GigaAM part

print("Load GigaAM transcriber")

import torchaudio
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.modules.audio_preprocessing import (
	AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
	FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)

import locale

locale.getpreferredencoding = lambda: "UTF-8"

class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
	def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
		if "window_size" in kwargs:
			del kwargs["window_size"]
		if "window_stride" in kwargs:
			del kwargs["window_stride"]

		super().__init__(**kwargs)

		self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
			torchaudio.transforms.MelSpectrogram(
				sample_rate=self._sample_rate,
				win_length=self.win_length,
				hop_length=self.hop_length,
				n_mels=kwargs["nfilt"],
				window_fn=self.torch_windows[kwargs["window"]],
				mel_scale=mel_scale,
				norm=kwargs["mel_norm"],
				n_fft=kwargs["n_fft"],
				f_max=kwargs.get("highfreq", None),
				f_min=kwargs.get("lowfreq", 0),
				wkwargs=wkwargs,
			)
		)


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
	def __init__(self, mel_scale: str = "htk", **kwargs):
		super().__init__(**kwargs)
		kwargs["nfilt"] = kwargs["features"]
		del kwargs["features"]
		self.featurizer = (
			FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
				mel_scale=mel_scale,
				**kwargs,
			)
		)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EncDecRNNTBPEModel.from_config_file("./rnnt_model_config.yaml")
ckpt = torch.load("./rnnt_model_weights.ckpt", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.eval()
model = model.to(device)

#import time
#start_t = time.time()
#print(model.transcribe(["./long_example.wav"])[0])
#print("Inference time:",time.time()-start_t)

@app.route("/transcribe", methods=['POST'])
def transcribeFunction():
	start_t = time.time()
	data = request.data
	if data is None or len(data)<4000:
		res = dict()
		res['text'] = ''
		res['error'] = 'There is no file posted'
		res['time'] = time.time()-start_t
		return res
	with tempfile.NamedTemporaryFile(suffix=".wav") as fl:
		fl.write(data)
		fl.flush()
		txt = model.transcribe([fl.name])[0]
		if type(txt)==list:
			txt = txt[0]
		res = dict()
		res['text'] = txt
		res['time'] = time.time()-start_t
		return res

##################### TRANSLATOR PART

enAlph = 'qwertyuiopasdfghjklzxcvbnm'
ruAlph = 'йцукенгшщзхъфывапролджэячсмитьбю'
def isItEnglish(txt):
	txtl = txt.lower()
	ruScore = sum([i in txtl for i in ruAlph])
	enScore = sum([i in txtl for i in enAlph])
	return enScore>=ruScore

#print("Load ru->en translator")
#
#
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#ruen_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
#ruen_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
#
#from transformers import FSMTForConditionalGeneration, FSMTTokenizer
#mname = "facebook/wmt19-ru-en"
#ruen_tokenizer2 = FSMTTokenizer.from_pretrained(mname)
#ruen_model2 = FSMTForConditionalGeneration.from_pretrained(mname)
#
#def ru2en(txt):
#	input_ids = ruen_tokenizer.encode(txt.replace("ё","е").replace("Ё","Е"), return_tensors="pt")
#	outputs = ruen_model.generate(input_ids)
#	decoded = ruen_tokenizer.decode(outputs[0], skip_special_tokens=True)
#	if len(decoded.replace(" ",""))==0 and len(txt.replace(" ",""))>0:
#		print("ru2en: Не удалось выполнить перевод промпта: "+txt)
#		return txt
#	return decoded
#
#def ru2en_v2(txt):
#	input_ids = ruen_tokenizer2.encode(txt.replace("ё","е").replace("Ё","Е"), return_tensors="pt")
#	outputs = ruen_model2.generate(input_ids)
#	decoded = ruen_tokenizer2.decode(outputs[0], skip_special_tokens=True)
#	if len(decoded.replace(" ",""))==0 and len(txt.replace(" ",""))>0:
#		print("ru2en: Не удалось выполнить перевод промпта: "+txt)
#		return txt
#	return decoded
#
#@app.route("/ru2en", methods=['GET','POST'])
#def translateFunction():
#	start_t = time.time()
#	txt = request.args.get('text')
#	res = dict()
#	try:
#		if isItEnglish(txt):
#			res['text'] = txt # Nothing do
#			res['text2'] = txt # Nothing do
#			res['error'] = "Text already in English, nothing do"
#		else:
#			res['text'] = ru2en(txt)
#			res['text2'] = ru2en_v2(txt)
#	except Exception as e:
#		res['text'] = ""
#		res['error'] = str(e)
#	res['time'] = time.time()-start_t
#	return res

####################################### CLIP encoder

print("Load CLIP model")
from PIL import Image
#from transformers import CLIPProcessor, CLIPModel

#model_name = "openai/clip-vit-large-patch14" # dim=768
#model_name = "openai/clip-vit-base-patch32" # dim=512

#clip_model = CLIPModel.from_pretrained(model_name)
#clip_processor = CLIPProcessor.from_pretrained(model_name)

import ruclip
clip_model, clip_processor = ruclip.load("ruclip-vit-base-patch32-384", device="cpu")
clip_predictor = ruclip.Predictor(clip_model, clip_processor, device="cpu", bs=8,quiet=True)

def img2vec(img):
	if type(img)==str:
		if img.startswith('http'):
			img = Image.open(requests.get(img, stream=True).raw)
		else:
			img = Image.open(img)
	if max(img.size)>1024: # So big
		f = max(img.size)/1024
		w,h = round(img.size[0]/f),round(img.size[1]/f)
		img = img.resize((w,h))
	res = dict()
	with torch.no_grad():
		res['clip'] = clip_predictor.get_image_latents([img])[0]
	return res

def text2vec(txt):
	if len(txt)<2: # Пустой запрос
		print("Пустой промпт")
		return np.zeros(512)
	with torch.no_grad():
		return clip_predictor.get_text_latents([txt]).numpy()[0]

@app.route("/clip_text_encode", methods=['GET','POST'])
def CLIP_text_encoder():
	start_t = time.time()
	txt = request.args.get('text')
#	translator_id = int(request.args.get('translator',0))
	res = dict()
	try:
#		if isItEnglish(txt):
#			res['text'] = txt # Nothing do
#		else:
#			if translator_id==0:
#				res['text'] = ru2en(txt)
#			elif translator_id==1:
#				res['text'] = ru2en_v2(txt)
#			else:
#				raise Exception("Translator not supported")
		emb = text2vec(txt)
#		emb = (emb/np.linalg.norm(emb))
		res['embedding'] = emb.tolist()
	except Exception as e:
		res['embedding'] = []
		res['error'] = str(e)
	res['time'] = time.time()-start_t
	return res

@app.route("/clip_img_encode", methods=['POST'])
def CLIP_img_encoder():
	start_t = time.time()
	data = request.data
	if data is None or len(data)<100:
		res = dict()
		res['error'] = 'There is no file posted'
		res['time'] = time.time()-start_t
		return res
	with tempfile.NamedTemporaryFile(suffix=".jpg") as fl:
		fl.write(data)
		fl.flush()
		emb = img2vec(fl.name)['clip']
		res = dict()
		res['embedding'] = emb.tolist()
		res['time'] = time.time()-start_t
		return res

# doc2vec encoding

print("doc2vec encoding")

import pymorphy3

morph = pymorphy3.MorphAnalyzer()

def getStartForm(w):
	try:
		return morph.parse(w)[0].normal_form
	except:
		return w


from sentence_transformers import SentenceTransformer
docvec_model = SentenceTransformer('cointegrated/LaBSE-en-ru')

@app.route("/doc2vec", methods=['GET','POST'])
def doc2vec_handle():
	start_t = time.time()
	txt = request.args.get('text')
	res = dict()
	try:
		emb = docvec_model.encode(txt).tolist()
		res['embedding'] = emb
		res['isEn'] = isItEnglish(txt)
		if not res['isEn']:
			res['lemmed'] = [getStartForm(w) for w in txt.split(" ")]
	except Exception as e:
		res['embedding'] = []
		res['lemmed'] = []
		res['isEn'] = False
		res['error'] = str(e)
	res['time'] = time.time()-start_t
	return res

# Video processing

print("Define video processing features")

import subprocess,os
from sklearn.cluster import KMeans
import cv2
import numpy as np

def opencvFrame2embedding(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # -> np RGB
	image = Image.fromarray(image) # -> PILlow
	return img2vec(image)['clip'].tolist() # -> Embedding

stop_words = ["поставь лайк этому ролику и подпишись на канал",
			"поставь лайк этому видео и подпишись на канал",
			"поставь лайк этому ролику и подпишись на меня",
			"поставь лайк этому видео и подпишись на меня",
			"поставь лайк этому ролику и подпишись",
			"поставь лайк этому видео и подпишись",
			"поставь лайк и подпишись на мой канал",
			"поставь лайк и подпишись на канал",
			"ссылка на мой дзен канал в шапке моего профиля",
			"не забудь поставить лайк и подписаться на канал",
			"поставь лайк и подпишись",
			"ставь лайк если было полезно",
			"а с тебе лайк и подписка",
			"с тебе лайк и подписка",
			"а с тебя лайк и подписка",
			"с тебя лайк и подписка",
			"все ссылки в описании",
			"ссылка в описании",
			"пиши в комментариях какой",
			"пиши в комментариях какая",
			"пиши в комментариях какой",
			"перед началом лайк подписка",
			"название в конце видео",
			"но вы про лайк и подписку не забудьте",
			"перед началом поставьте лайк подпишитесь",
			"вы про лайк и подписку не забудьте",
			"про лайк и подписку не забудьте",
			"ставь лайк и подписывайся если",
			"ставь лайк и подписывайся",
			"подпишись и поставь лайк",
			"не забудь подписаться и поставить лайк",
			"номер в комментариях чтобы не потерялся",
			"кстати не забудьте подписаться на мой блог",
			"не забудьте подписаться на мой блог",
			"лайкнуть этот расклад",
			"я не чувствую ваши лайки",
			"переслать этот пост другим подругам",
			"ставь лайк"
		]

stop_words.sort(key=lambda x:-len(x))

def getBaseIndices(clusters):
	classes = {i:[] for i in set(clusters)}
	for i,val in enumerate(clusters):
		if i==0:
			classes[val].append(1)
		else:
			classes[val].append(classes[val][-1]+1)
		for k in classes.keys():
			if k==val:continue
			classes[k].append(0)
	best_inds = []
	for cl in classes.keys():
		max_val = max(classes[cl])
		best_ind = classes[cl].index(max_val)-max_val//2
		best_inds.append(best_ind)
	return best_inds

def descriptionFilter(description,thres=5):
	tmp = description.count("#втоп")+description.count("#врекомендации")
	tmp += description.count("#boobs")+description.count("#bigass")+description.count("#pussy")+description.count("#ass")
	if description.count("#")>thres:
		description = " ".join([w for w in description.split(" ") if not w.startswith("#")])
	return description

def video2dict(videopath,description="",stride=5,K=5):
	tmpfl = None
	result = dict()
	if videopath.startswith("http"):
		tmpfl = tempfile.NamedTemporaryFile(suffix=".mp4")
		r = requests.get(videopath)
		tmpfl.write(r.content)
		tmpfl.flush()
		videopath=tmpfl.name
	# Audio processing
	command = "ffmpeg -i "+videopath+" -ac 1 -vn "+videopath+".wav"
	subprocess.call(command, shell=True)
	try:
		audio_transcription = model.transcribe([videopath+".wav"])[0][0]
		for stpwrd in stop_words:
			audio_transcription = audio_transcription.replace(stpwrd,"")
		audio_transcription = audio_transcription.replace("  "," ")
		result['transcription'] = audio_transcription
		if len(audio_transcription)>0:
			result['transcription_embedding'] = docvec_model.encode(audio_transcription)
		else:
			result['transcription_embedding'] = np.zeros(768)
		result['transcription_lemmed'] = [getStartForm(w) for w in audio_transcription.split(" ")]
		os.remove(videopath+".wav")
	except:
		print("Transcription error:",videopath)
		result['transcription'] = ""
		result['transcription_lemmed'] = []
		result['transcription_embedding'] = np.zeros(768)
		os.remove(videopath+".wav")
	result['description_orig'] = description
	result['description'] = descriptionFilter(description,thres=6)
	result['description_lemmed'] = result['description'].replace(","," ").replace("  "," ").replace("  "," ").replace("  "," ")
	result['description_lemmed'] = [getStartForm(w) for w in result['description_lemmed'].split(" ") ]
	result['description_embedding'] = docvec_model.encode( result['description'] )
	# Image processing: frames extracting
	vidcap = cv2.VideoCapture(videopath)
	frames = []
	success,image = vidcap.read()
	frames.append( opencvFrame2embedding(image) )
	count = 0
	while success:
		success,image = vidcap.read()
		count += 1
		if success and count%stride==0:
			frames.append( opencvFrame2embedding(image) )
	print(count,"frames")
	# Image processing: find keyframes
	frames = np.array(frames)
	kmodel = KMeans(n_clusters=K)
	kmodel.fit(frames)
	clusters = kmodel.predict(frames)
	keyframes_inds = getBaseIndices(clusters)
	result['keyframes'] = [i*stride for i in keyframes_inds]
	result['clip_embeddings'] = frames[keyframes_inds]
	vidcap.release()
	tmpfl.close()
	return result

@app.route("/encodeVideoByLink", methods=['GET','POST'])
def video2vec_handle():
	start_t = time.time()
	url = request.args.get('url')
	description = request.args.get('description',"")
	stride = int(request.args.get('stride',5))
	K = int(request.args.get('K',5))
	res = dict()
	try:
		emb = video2dict(url,description=description,stride=stride,K=K)
		for k in emb.keys():
			if isinstance(emb[k],np.ndarray):
				emb[k] = emb[k].tolist()
		res['result'] = emb
	except Exception as e:
		res['result'] = {}
		res['error'] = str(e)
	res['time'] = time.time()-start_t
	return res

if __name__ == "__main__":
	app.run(host='127.0.0.1',debug=False,port=6000)