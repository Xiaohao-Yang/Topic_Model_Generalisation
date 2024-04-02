import ot
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from scipy.spatial.distance import cosine
import math


# Compute HOT
def doc_hot(theta_original, theta_aug, beta, word_embedding, topk=10):
    # select only top k words
    top_idx_topic = torch.argsort(beta, descending=True)[:, 0:topk]
    top_idx_topic = torch.sort(top_idx_topic)[0]
    topic = torch.gather(beta, 1, top_idx_topic)
    topic = torch.div(topic, topic.sum(dim=-1).unsqueeze(-1))

    # construct cost matrix
    doc_M = torch.zeros(size=(topic.shape[0], topic.shape[0]))
    word_embedding = torch.from_numpy(word_embedding).to(torch.float64).cuda()
    for i in range(topic.shape[0]):
        for j in range(topic.shape[0]):
            if i == j:
                doc_M[i, j] = 0.
            else:
                word_embedding_k1 = word_embedding[top_idx_topic[i], :]
                word_embedding_k2 = word_embedding[top_idx_topic[j], :]
                M = 1 - pairwise_cosine_similarity(word_embedding_k1, word_embedding_k2)
                doc_M[i, j] = ot.emd2(topic[i], topic[j], M)
    doc_M = doc_M.to(torch.float64)

    # compute Sinkhorn distance
    doc_dist = sinkhorn_torch(doc_M.cuda(), theta_original.T, theta_aug.T).mean()

    return doc_dist


# Sinkhorn distance
def sinkhorn_torch(M, a, b, lambda_sh=100, numItermax=5000, stopThr=.5e-2, cuda=True):
    if cuda:
        u = (torch.ones_like(a) / a.size()[0]).double().cuda()
        v = (torch.ones_like(b)).double().cuda()
    else:
        u = (torch.ones_like(a) / a.size()[0])
        v = (torch.ones_like(b))

    K = torch.exp(-M * lambda_sh)
    err = 1
    cpt = 0
    while err > stopThr and cpt < numItermax:
        u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t())))
        cpt += 1
        if cpt % 20 == 1:
            v = torch.div(b, torch.matmul(K.t(), u))
            u = torch.div(a, torch.matmul(K, v))
            bb = torch.mul(v, torch.matmul(K.t(), u))
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

    sinkhorn_divergences = torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0)
    return sinkhorn_divergences


# Consine distance
def avg_cosine_d(theta1, theta2):
    n_doc = theta1.shape[0]

    total = 0
    for i in range(n_doc):
        dist = cosine(theta1[i], theta2[i])
        total += dist
    return total/n_doc


def hellinger_fast(p, q):
    return sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p,q) ])


# Hellinger distance
def avg_hellinger(P,Q):
    dists = torch.ones(P.shape[0]).cuda()
    for i in range(P.shape[0]):
        dists[i] = hellinger_fast(P[i], Q[i])

    return dists.mean()