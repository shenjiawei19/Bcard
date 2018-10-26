# from data_ready import get_data
# import math

# dictionary, similarity_matrix, tfidf = get_data()


# def similar_part(x, y):
#     sim_group = []
#     for t in x:
#         res = [similarity_matrix[t[0], j[0]] for j in y]
#         if len(res) > 0:
#             sim_group.append(max(res))
#     return math.exp(5*(sum(sim_group)/len(x)))


# def diff_part(x, y):
#     sim_group = []
#     for t in x:
#         res = [similarity_matrix[t[0], j[0]] for j in y]
#         if len(res) > 0:
#             sim_group.append(max(res))
#     for t in y:
#         res = [similarity_matrix[t[0], j[0]] for j in x]
#         if len(res) > 0:
#             sim_group.append(max(res))
#     return math.exp(5*((len(x)+len(y) - sum(sim_group))/(len(x)+len(y))))


# def calculate(x, y):
#     sim_part = similar_part(x, y)
#     de_part = diff_part(x, y)
#     return sim_part/(sim_part+de_part)


# def new_method(query, documents):
#     query = tfidf[dictionary.doc2bow(query)]
#     similarities = [calculate(
#         query, tfidf[dictionary.doc2bow(document)]) for document in documents]
#     return similarities


# from data_ready import get_data
# import math

# dictionary, similarity_matrix, tfidf = get_data()


# def similar_part(x, y):
#     sim_group = []
#     index = 0
#     for t in x:
#         print(x.index(t))
#         z = y[(index-5):(index+5)]
#         res = [similarity_matrix[t[0], j[0]] for j in z]
#         if len(res) > 0:
#             sim_group.append(max(res))
#         index += 1
#     return math.exp(100*(sum(sim_group)/len(x)))


# def diff_part(x, y):
#     sim_group = []
#     index = 0
#     for t in x:
#         z = y[index-5:index+5]
#         res = [similarity_matrix[t[0], j[0]] for j in z]
#         if len(res) > 0:
#             sim_group.append(max(res))
#             index += 0
#     index = 0
#     for t in y:
#         z = x[index-5:index + 5]
#         res = [similarity_matrix[t[0], j[0]] for j in z]
#         if len(res) > 0:
#             sim_group.append(max(res))
#         index += 1
#     return math.exp(100*((len(x)+len(y) - sum(sim_group))/(len(x)+len(y))))


# def calculate(x, y):
#     sim_part = similar_part(x, y)
#     de_part = diff_part(x, y)
#     return sim_part/(sim_part+de_part)


# def new_method(query, documents):
#     query = tfidf[dictionary.doc2bow(query)]
#     similarities = [calculate(
#         query, tfidf[dictionary.doc2bow(document)]) for document in documents]
#     return similarities


from data_ready import get_data
import math

dictionary, similarity_matrix, tfidf = get_data()


def similar_part(x, y):
    sim_group = []
    for t in x:
        res = [similarity_matrix[t[0], j[0]] for j in y]
        if len(res) > 0:
            sim_group.append(max(res))
    return sum(sim_group)


def diff_part(x, y):
    sim_group = []
    for t in x:
        res = [similarity_matrix[t[0], j[0]] for j in y]
        if len(res) > 0:
            sim_group.append(max(res))
    for t in y:
        res = [similarity_matrix[t[0], j[0]] for j in x]
        if len(res) > 0:
            sim_group.append(max(res))
    return sum(sim_group)


def calculate(x, y):

    sim_part = similar_part(x, y)
    de_part = diff_part(x, y)

    if sim_part > de_part:
        sim_part = math.exp(0.5*(sim_part/len(x)))
        de_part = math.exp(((len(x)+len(y) - de_part)/(len(x)+len(y))))
    else:
        sim_part = math.exp((sim_part/len(x)))
        de_part = math.exp(0.5*((len(x)+len(y) - de_part)/(len(x)+len(y))))

    return sim_part/(sim_part+de_part)


def new_method(query, documents):
    query = tfidf[dictionary.doc2bow(query)]
    similarities = [calculate(
        query, tfidf[dictionary.doc2bow(document)]) for document in documents]
    return similarities
