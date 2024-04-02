import numpy as np


def make_network(options, vocab_size, word_embedding, label_type=None, n_labels=0, n_prior_covars=0, n_topic_covars=0):
    # Assemble the network configuration parameters into a dictionary
    network_architecture = \
        dict(embedding_dim=options.emb_dim,
             n_topics=options.n_topic,
             vocab_size=vocab_size,
             word_embedding=word_embedding,
             label_type=label_type,
             n_labels=n_labels,
             n_prior_covars=n_prior_covars,
             n_topic_covars=n_topic_covars,
             l1_beta_reg=options.l1_topics,
             l1_beta_c_reg=options.l1_topic_covars,
             l1_beta_ci_reg=options.l1_interactions,
             l2_prior_reg=options.l2_prior_covars,
             classifier_layers=1,
             use_interactions=options.interactions,
             dist=options.dist,
             reg_weight = options.reg_weight,
             aug_method = options.DA,
             reg = options.regularisation
             )
    return network_architecture


def get_init_bg(data):
    #Compute the log background frequency of all words
    #sums = np.sum(data, axis=0)+1
    n_items, vocab_size = data.shape
    sums = np.array(data.sum(axis=0)).reshape((vocab_size,))+1.
    print("Computing background frequencies")
    print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg


def create_minibatch(X, Y, PC=None, TC=None, batch_size=None, rng=None):
    # Yield a random minibatch
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            ixs = rng.randint(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)

        X_mb = np.array(X[ixs, :]).astype('float32')
        if Y is not None:
            Y_mb = Y[ixs, :].astype('float32')
        else:
            Y_mb = None

        if PC is not None:
            PC_mb = PC[ixs, :].astype('float32')
        else:
            PC_mb = None

        if TC is not None:
            TC_mb = TC[ixs, :].astype('float32')
        else:
            TC_mb = None

        yield X_mb, Y_mb, PC_mb, TC_mb
