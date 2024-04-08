import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from operator import itemgetter
import math
from scipy.special import softmax as softmax_np


# random forest for ACC
def rf_cls(train_theta, train_y, test_theta, test_y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=0)

    train_theta = train_theta.astype('float32')
    test_theta = test_theta.astype('float32')
    train_y = train_y.ravel()
    test_y = test_y.ravel()

    clf.fit(train_theta, train_y)
    predict_test = clf.predict(test_theta)
    acc = metrics.accuracy_score(test_y, predict_test)

    return acc


# top-Purity; top-NMI
def evaluate_TP_TN(label, theta):
    golden_clusters = {}
    output_clusters = {}
    num_docs = len(label)

    # Create golden clusters
    for id, lbl in enumerate(label):
        ids = golden_clusters.get(lbl, set())
        ids.add(id)
        golden_clusters[lbl] = ids

    # Create output clusters
    max_topic = np.argmax(theta, axis=1)
    for doc_id, topic_id in enumerate(max_topic):
        lbl = "Topic_" + str(topic_id)
        ids = output_clusters.get(lbl, set())
        ids.add(doc_id)
        output_clusters[lbl] = ids

    # Compute purity
    count = 0
    for _, docs in output_clusters.items():
        correct_assigned_doc_num = 0
        for _, golden_docs in golden_clusters.items():
            correct_assigned_doc_num = max(correct_assigned_doc_num, len(docs.intersection(golden_docs)))
        count += correct_assigned_doc_num
    purity = count / num_docs

    # Compute NMI
    MI_score = 0.0
    for _, docs in output_clusters.items():
        for _, golden_docs in golden_clusters.items():
            num_correct_assigned_docs = len(docs.intersection(golden_docs))
            if num_correct_assigned_docs == 0.0:
                continue
            MI_score += (num_correct_assigned_docs / num_docs) * math.log(
                (num_correct_assigned_docs * num_docs) / (len(docs) * len(golden_docs)))
    entropy = 0.0
    for _, docs in output_clusters.items():
        entropy += (-1.0 * len(docs) / num_docs) * math.log(1.0 * len(docs) / num_docs)
    for _, docs in golden_clusters.items():
        entropy += (-1.0 * len(docs) / num_docs) * math.log(1.0 * len(docs) / num_docs)
    NMI = 2 * MI_score / entropy

    return purity, NMI


def print_topics(beta, voc, save_file, topn=10):
    K = beta.shape[0]
    with open(save_file, 'w') as file:
        for i in range(K):
            top_word_idx = np.argsort(beta[i, :])[::-1]
            top_word_idx = top_word_idx[0:topn]
            top_words = itemgetter(*top_word_idx)(voc)
            for item in top_words:
                file.write(item + ' ')
            file.write('\n')
        file.close()


# document completion perplexity
def perplexity(model, test_1, test_2, beta):
    model.eval()
    theta_test1 = model.get_theta(test_1).cpu().detach().numpy()
    sums_2 = test_2.data.sum(1)

    preds = np.log(np.matmul(theta_test1, beta))
    recon_loss = -(preds * test_2.data).sum(1)
    loss = recon_loss / (sums_2 + 1e-10)
    loss = np.nanmean(loss)
    ppl = round(math.exp(loss), 1)

    return ppl


def get_doc_rep_source(model, name, data_dict):
    if name in ['CLNTM', 'SCHOLAR']:
        train_theta = model.get_theta(data_dict['train_data'], data_dict['train_label'])
        test_theta = model.get_theta(data_dict['test_data'], data_dict['test_label'])
    else:
        train_theta = model.get_theta(data_dict['train_data']).cpu().detach().numpy()
        test_theta = model.get_theta(data_dict['test_data']).cpu().detach().numpy()

    return train_theta, test_theta


def get_doc_rep_target(model, name, data_dict):
    train_theta_targets = []
    test_theta_targets = []
    if isinstance(data_dict['train_data_target'], list):  # if multiple targets
        for i in range(len(data_dict['train_data_target'])):
            if name in ['CLNTM', 'SCHOLAR']:
                train_theta_target = model.get_theta(data_dict['train_data_target'][i],
                                                     data_dict['train_label_target'][i])
                test_theta_target = model.get_theta(data_dict['test_data_target'][i],
                                                    data_dict['test_label_target'][i])
            else:
                train_theta_target = model.get_theta(data_dict['train_data_target'][i]).cpu().detach().numpy()
                test_theta_target = model.get_theta(data_dict['test_data_target'][i]).cpu().detach().numpy()

            train_theta_targets.append(train_theta_target)
            test_theta_targets.append(test_theta_target)

    else:  # only one target
        if name in ['CLNTM', 'SCHOLAR']:
            train_theta_target = model.get_theta(data_dict['train_data_target'], data_dict['train_label_target'])
            test_theta_target = model.get_theta(data_dict['test_data_target'], data_dict['test_label_target'])
        else:
            train_theta_target = model.get_theta(data_dict['train_data_target']).cpu().detach().numpy()
            test_theta_target = model.get_theta(data_dict['test_data_target']).cpu().detach().numpy()

        train_theta_targets.append(train_theta_target)
        test_theta_targets.append(test_theta_target)

    return train_theta_targets, test_theta_targets


def evaluation(model, epoch, data_dict, args):
    model.eval()
    parameter_setting = '%s_dataset:%s_K%s_RS%s_epochs:%s_LR%s_reg:%s_regW%s_augRate:%s_aug:%s' \
                        % (args.name, args.dataset, args.n_topic, args.seed, epoch, args.lr,
                           args.regularisation, args.reg_weight, args.aug_rate, args.DA)
    print('############################################')
    print('Evaluation at: ')
    print(parameter_setting)
    print()

    # save topics, for further evaluation of topic coherence
    # save_dir = None # define your own path
    # save_file_topics = os.path.join(save_dir, 'vis.txt')
    emb = model.get_beta().cpu().detach().numpy()
    # print_topics(emb, data_dict['idx2token'], save_file_topics)

    # topic-word distribution
    emb = softmax_np(emb, -1)

    # document representations
    train_theta, test_theta = get_doc_rep_source(model, args.name, data_dict)
    train_theta_targets, test_theta_targets = get_doc_rep_target(model, args.name, data_dict)

    # source doc acc evaluation
    source_acc = rf_cls(train_theta, data_dict['train_label'], test_theta, data_dict['test_label'])

    # target doc acc evaluation
    if len(train_theta_targets) > 1: # if multiple targets
        target_accs = []
        for i in range(len(train_theta_targets)):
            target_acc = rf_cls(train_theta_targets[i], data_dict['train_label_target'][i],
                                test_theta_targets[i], data_dict['test_label_target'][i])
            target_accs.append(target_acc)

        print('doc classification acc (original corpus): ', source_acc)
        print('doc classification acc (R8): ', target_accs[0])
        print('doc classification acc (DBpedia): ', target_accs[1])
        print('doc classification acc (TMN): ', target_accs[2])
        print('doc classification acc (Webs): ', target_accs[3])

    else: # only one target
        target_acc = rf_cls(train_theta_targets[0], data_dict['train_label_target'],
                            test_theta_targets[0], data_dict['test_label_target'])

        print('doc classification acc (original corpus): ', source_acc)
        print('doc classification acc (target corpus): ', target_acc)

    # source doc clustering evaluation
    print('############################################')
    TP_source, TN_source = evaluate_TP_TN(np.squeeze(data_dict['test_label']), test_theta)

    # target doc clustering evaluation
    if len(test_theta_targets) > 1: # if multiple targets
        TP_targets = []
        TN_targets = []
        for i in range(len(test_theta_targets)):
            TP_target, TN_target = evaluate_TP_TN(np.squeeze(data_dict['test_label_target'][i]), test_theta_targets[i])
            TP_targets.append(TP_target)
            TN_targets.append(TN_target)

        print('doc clustering TP, TN (original corpus): ', TP_source, TN_source)
        print('doc clustering TP, TN (R8): ', TP_targets[0], TN_targets[0])
        print('doc clustering TP, TN (DBpedia): ', TP_targets[1], TN_targets[1])
        print('doc clustering TP, TN (TMN): ', TP_targets[2], TN_targets[2])
        print('doc clustering TP, TN (Webs): ', TP_targets[3], TN_targets[3])

    else:
        TP_target, TN_target = evaluate_TP_TN(np.squeeze(data_dict['test_label_target']), test_theta_targets[0])

        print('doc clustering TP, TN (original corpus): ', TP_source, TN_source)
        print('doc clustering TP, TN (target corpus): ', TP_target, TN_target)

    # document completion perplexity
    if not args.name in ['CLNTM', 'SCHOLAR']:
        ppl = perplexity(model, data_dict['test1_data'], data_dict['test2_data'], emb)
    else:
        ppl = model.perplexity(data_dict['test1_data'], data_dict['test_1_label'], data_dict['test2_data'], emb)

    print('############################################')
    print('source document completion ppl: ', ppl)
    print('############################################')
