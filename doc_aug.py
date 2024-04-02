import json
import copy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


def create_similarity_dict(word_embedding, save_path, topn=10):
    similarity_dict = {}
    dissimilarity_dict = {}
    for i in range(word_embedding.shape[0]):
        similarity = cosine_similarity(word_embedding[i].reshape(1, -1), word_embedding).tolist()[0]
        similarity_tuple = [(idx, s) for idx, s in enumerate(similarity)]

        similarity_tuple_sorted = sorted(similarity_tuple, key=lambda tup: tup[1], reverse=True)[1:topn+1]
        similarity_dict[i] = [t[0] for t in similarity_tuple_sorted]

        dissimilarity_tuple_sorted = sorted(similarity_tuple, key=lambda tup: tup[1])[0:topn]
        dissimilarity_dict[i] = [t[0] for t in dissimilarity_tuple_sorted]

    with open('%s/similar.json' % save_path, 'w') as file:
        file.write(json.dumps(similarity_dict))
        file.close()
    with open('%s/dissimilar.json' % save_path, 'w') as file:
        file.write(json.dumps(dissimilarity_dict))
        file.close()


def bow2list(bow):
    doc_list = []
    for idx, freq in enumerate(bow):
        if freq > 0:
            for j in range(int(freq)):
                doc_list.append(idx)
    return doc_list


def get_tfidf(train_data):
    # get tf-idf weights
    N = train_data.shape[0]
    df = copy.deepcopy(train_data)
    df[torch.where(df > 0)] = 1
    df = torch.sum(df, 0)
    idf = torch.log((N + 1) / (1 + df))
    tfidf = train_data * idf

    return tfidf


# random similar
def random_similar(train_data, similarity_dict, dissimilarity_dict, replace_proportion=0.2, mode='similar'):
    for i in range(train_data.shape[0]):
        L = train_data[i].sum()
        num_replace = int(torch.floor(torch.tensor(L * replace_proportion)))

        # convert bow to list of index
        doc_list = bow2list(train_data[i])
        # get the words that want to deal with
        selected_idx = np.random.choice(doc_list, num_replace, replace=False)

        # replace
        for j in range(num_replace):
            try:
                if mode == 'similar':
                    replace_word = np.random.choice(similarity_dict[str(selected_idx[j].item())])
                else:
                    replace_word = np.random.choice(dissimilarity_dict[str(selected_idx[j].item())])

                train_data[i, selected_idx[j]] -= 1
                train_data[i, replace_word] += 1

            except:
                pass

    return train_data


# only highest -- similar
def high_similar(train_data, aug_rate, similarity_dict):
    if not torch.is_tensor(train_data):
        train_data = torch.from_numpy(train_data)
    tfidf = get_tfidf(train_data)

    # for each doc in batch
    for i in range(train_data.shape[0]):
        L = torch.sum(train_data[i])
        num_replace = int(torch.floor(L * aug_rate))

        # get the words that want to deal with
        nonzero_idx = torch.where(tfidf[i] > 0)[0]
        nonzero_value = tfidf[i][nonzero_idx]
        idx_weight_tuple = [(idx, value) for idx, value in zip(nonzero_idx, nonzero_value)]

        # important words
        idx_weight_tuple_important = sorted(idx_weight_tuple, key=lambda tup: tup[1], reverse=True)
        top_word_idx_important = [item[0] for item in idx_weight_tuple_important[0:num_replace]]

        # replace
        for j in range(num_replace):
            try:
                replace_important = np.random.choice(similarity_dict[str(top_word_idx_important[j].item())])
                train_data[i, top_word_idx_important[j]] -= 1
                train_data[i, replace_important] += 1
            except:
                pass

    return train_data


# lowest -- similar
def low_similar(train_data, aug_rate, similarity_dict):
    tfidf = get_tfidf(train_data)

    # for each doc in batch
    for i in range(train_data.shape[0]):
        L = torch.sum(train_data[i])
        num_replace = int(torch.floor(L * aug_rate))

        # get the words that want to deal with
        nonzero_idx = torch.where(tfidf[i] > 0)[0]
        nonzero_value = tfidf[i][nonzero_idx]
        idx_weight_tuple = [(idx, value) for idx, value in zip(nonzero_idx, nonzero_value)]

        # important words and unimportant words
        idx_weight_tuple_unimportant = sorted(idx_weight_tuple, key=lambda tup: tup[1])
        top_word_idx_unimportant = [item[0] for item in idx_weight_tuple_unimportant[0:num_replace]]

        # replace
        for j in range(num_replace):
            try:
                replace_unimportant = np.random.choice(similarity_dict[str(top_word_idx_unimportant[j].item())])
                train_data[i, top_word_idx_unimportant[j]] -= 1
                train_data[i, replace_unimportant] += 1
            except:
                pass

    return train_data


# highest -- similar; lowest to similar
def high_low_similar(train_data, aug_rate, similarity_dict):
    aug_rate = aug_rate/2
    tfidf = get_tfidf(train_data)

    # for each doc in batch
    for i in range(train_data.shape[0]):
        L = torch.sum(train_data[i])
        num_replace = int(torch.floor(L * aug_rate))

        # get the words that want to deal with
        nonzero_idx = torch.where(tfidf[i] > 0)[0]
        nonzero_value = tfidf[i][nonzero_idx]
        idx_weight_tuple = [(idx, value) for idx, value in zip(nonzero_idx, nonzero_value)]

        # important words and unimportant words
        idx_weight_tuple_important = sorted(idx_weight_tuple, key=lambda tup: tup[1], reverse=True)
        idx_weight_tuple_unimportant = sorted(idx_weight_tuple, key=lambda tup: tup[1])
        top_word_idx_important = [item[0] for item in idx_weight_tuple_important[0:num_replace]]
        top_word_idx_unimportant = [item[0] for item in idx_weight_tuple_unimportant[0:num_replace]]

        # replace
        for j in range(num_replace):
            try:
                replace_important = np.random.choice(similarity_dict[str(top_word_idx_important[j].item())])
                replace_unimportant = np.random.choice(similarity_dict[str(top_word_idx_unimportant[j].item())])

                train_data[i, top_word_idx_important[j]] -= 1
                train_data[i, replace_important] += 1

                train_data[i, top_word_idx_unimportant[j]] -= 1
                train_data[i, replace_unimportant] += 1
            except:
                pass

    return train_data