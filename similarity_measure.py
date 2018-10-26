import logging
import math
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from math import isnan
from time import time
from gensim.similarities import MatrixSimilarity, WmdSimilarity, SoftCosineSimilarity
import numpy as np
from sklearn.model_selection import KFold
from new_method import new_method
from jaccard import jacard_similarity, jacard_base
import gensim.downloader as api
from data_ready import preprocess
from data_ready import get_data
import math

dictionary, similarity_matrix, tfidf = get_data()
datasets = api.load("semeval-2016-2017-task3-subtaskBC")



def produce_test_data(dataset):
    for orgquestion in datasets[dataset]:
        query = preprocess(orgquestion["OrgQSubject"]) + \
            preprocess(orgquestion["OrgQBody"])
        documents = [
            preprocess(thread["RelQuestion"]["RelQSubject"]) +
            preprocess(thread["RelQuestion"]["RelQBody"])
            for thread in orgquestion["Threads"]]
        relevance = [
            thread["RelQuestion"]["RELQ_RELEVANCE2ORGQ"] in (
                "PerfectMatch", "Relevant")
            for thread in orgquestion["Threads"]]
        yield query, documents, relevance


def softcossim(query, documents):
    # Compute Soft Cosine Measure between the query and the documents.
    query = tfidf[dictionary.doc2bow(query)]
    index = SoftCosineSimilarity(
        tfidf[[dictionary.doc2bow(document) for document in documents]],
        similarity_matrix)
    similarities = index[query]
    return similarities

def cossim(query, documents):
    # Compute cosine similarity between the query and the documents.
    query = tfidf[dictionary.doc2bow(query)]
    index = MatrixSimilarity(
        tfidf[[dictionary.doc2bow(document) for document in documents]],
        num_features=len(dictionary))
    similarities = index[query]
    return similarities

strategies = {
    # "cossim": cossim,
    # "softcossim": softcossim,
    # "jacard_similarity": jacard_similarity
    "jacard_similarity": jacard_similarity
}


def evaluate(split, func, fun2=None):
    # Perform a single round of evaluation.
    results = []
    start_time = time()
    for query, documents, relevance in split:
        # similarities1 = func(query, documents)
        similarities = func(query, documents)
        # print(fun2)
        if fun2 is not None:
            similarities2 = fun2(query, documents)
            similarities = np.add(similarities, similarities2).tolist()
            # print(2)
            # for i in range(0, len(documents)):
            #     similarities.append(similarities[i]*0.5+similarities2[i]*0.5)

        assert len(similarities) == len(documents)
        precision = [
            (num_correct + 1) / (num_total + 1) for num_correct, num_total in enumerate(
                num_total for num_total, (_, relevant) in enumerate(
                    sorted(zip(similarities, relevance), reverse=True)) if relevant)]
        average_precision = np.mean(precision) if precision else 0.0
        results.append(average_precision)
    return (np.mean(results) * 100, time() - start_time)


def crossvalidate(*args):
    # Perform a cross-validation.
    dataset = args[0]
    strategy = args[1]
    strategy2 = args[2] if len(args) == 3 else None

    test_data = np.array(list(produce_test_data(dataset)))
    kf = KFold(n_splits=10)
    samples = []
    for _, test_index in kf.split(test_data):
        samples.append(evaluate(test_data[test_index], strategy, strategy2))
    return (np.mean(samples, axis=0), np.std(samples, axis=0))

if __name__ == '__main__':

    # args_list = ["2016-test", "jacard_similarity"]
    # args_list = ["2016-test", new_method]
    # args_list = ["2016-test", jacard_base]
    # args_list = ["2016-test", softcossim, jacard_base]
    # args_list = ["2016-test", softcossim, jacard_similarity]
    # args_list = ["2016-test", softcossim, new_method]

    # args_list = ["2017-test", new_method]
    # args_list = ["2017-test", jacard_similarity]
    # args_list = ["2017-test", jacard_base]
    args_list = ["2017-test", softcossim, jacard_base]
    results = crossvalidate(*args_list)
    print(results)
    args_list = ["2017-test", softcossim, jacard_similarity]
    results = crossvalidate(*args_list)
    print(results)
    args_list = ["2017-test", softcossim, new_method]
    results = crossvalidate(*args_list)
    print(results)

