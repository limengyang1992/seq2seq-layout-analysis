#coding=utf-8
# import gensim
from gensim.models.doc2vec import Doc2Vec
import linecache
def test_doc2vec():
    # 加载模型
    model = Doc2Vec.load('doc2vec/model')
    # 也可以推断一个句向量(未出现在语料中)
    count = len(open('4.txt', 'r', encoding='utf-8').readlines())  # 获取行数
    for i in range(0, count + 1):
        sentence = linecache.getline('4.txt', i)
        vec=model.infer_vector(sentence.split())
        print(sentence)
        print(vec)

if __name__ == '__main__':
    model = Doc2Vec.load('doc2vec/model')
    vec=model.infer_vector(list("好好学习天天向上"))
    print(len(vec))


