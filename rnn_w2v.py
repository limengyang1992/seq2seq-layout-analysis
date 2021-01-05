
from gensim.models import word2vec
import glob
import json
import csv
import random
import codecs

from config import Config
config = Config()
train_paths = glob.glob(config.train_jsons+"/*.json")
test_paths = glob.glob(config.test_jsons+"/*.json")
paths = train_paths + test_paths


raw_sentences = []

for path in paths:
    with open(path, 'rb') as f:
        data = json.loads(f.read())
        h = data["imageHeight"]
        w = data["imageWidth"]
        shapes = data["shapes"]
        for i, obj in enumerate(shapes):
            label = obj["label"].replace(" ","")
            raw_sentences.append([str(x).lower() for x in label if len(x)>0])

sentences = [s for s in raw_sentences if len(s)>0]
print(sentences)        
model = word2vec.Word2Vec(sentences, min_count=1)
 

vector = []
for each in model.wv.index2word:
    line=list(model[each])
    lines=[str(i) for i in line]
    linestr=' '.join(lines)
    L=each+' '+linestr
    vector.append(L)
vect='\n'.join(vector)
ff=codecs.open(config.w2v_path,'w+',encoding='utf-8')
ff.write(vect)
