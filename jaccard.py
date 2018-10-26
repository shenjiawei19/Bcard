from data_ready import get_data

dictionary, similarity_matrix, tfidf = get_data()


def jac_card(len_x, sim):
    sim = abs(sim)
    res = sim / (2 * len_x - sim) if (len_x > 0) else 0
    return res


def get_sim_max(x, y):
    res = [similarity_matrix[x[0], j[0]] for j in y]
    if len(res) > 0:
        return max(res)
    else:
        return 0


def jacard_similarity(x, y):
    x = tfidf[dictionary.doc2bow(x)]
    return [jac_card(len(x), sum(get_sim_max(t, tfidf[dictionary.doc2bow(i)]) for t in x)) for i in y]


def jacard_base(x, y):
    return [len((set(x) & set(d))) / len((set(x) | set(d))) for d in y]
