import gensim.downloader as api
from itertools import chain
import json
from re import sub
from os.path import isfile
import gensim.downloader as api
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk import download
import pandas as pd
import os
import pickle
from gensim.matutils import softcossim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from nltk.corpus import stopwords

# 下载数据

def preprocess(doc):

    stop_words = set(stopwords.words("english"))
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stop_words]


def processing_data():
    corpus = list(chain(*[
        chain(
            [preprocess(thread["RelQuestion"]["RelQSubject"]),
             preprocess(thread["RelQuestion"]["RelQBody"])],
            [preprocess(relcomment["RelCText"]) for relcomment in thread["RelComments"]])
        for thread in api.load("semeval-2016-2017-task3-subtaskA-unannotated")]))

    dictionary = Dictionary(corpus)
    tfidf = TfidfModel(dictionary=dictionary)
    w2v_model = Word2Vec(corpus, workers=cpu_count(),
                         min_count=5, size=300, seed=12345)
    similarity_matrix = w2v_model.wv.similarity_matrix(
        dictionary, tfidf, nonzero_limit=100)
    pickle.dump(dictionary, open(
        r'C:\Code\201810\Similarity\data\dic_path', 'wb+')) #字典
    pickle.dump(similarity_matrix, open(
        r'C:\Code\201810\Similarity\data\similarity_matrix_path', 'wb+')) #相似度举证
    pickle.dump(tfidf, open(
        r'C:\Code\201810\Similarity\data\tfidf_path', 'wb+')) #tfidf


def get_data():
    dictionary = pickle.load(
        open(r'C:\LNS\m_projects\Bcard\Similarity\data\dic_path', 'rb+'))
    similarity_matrix = pickle.load(
        open(r'C:\LNS\m_projects\Bcard\Similarity\data\similarity_matrix_path', 'rb+'))
    tfidf = pickle.load(
        open(r'C:\LNS\m_projects\Bcard\Similarity\data\tfidf_path', 'rb+'))

    return dictionary, similarity_matrix, tfidf
