from scipy import sparse
import scipy.io as sio


def sparse2dense(input_matrix):
    if sparse.isspmatrix(input_matrix):
        input_matrix = input_matrix.toarray()
    input_matrix = input_matrix.astype('float32')
    return input_matrix


# load original corpus
def load_train_data(mat_file_name, is_to_dense=True):
    data = sio.loadmat(mat_file_name)
    train_data = data['wordsTrain'].transpose()
    train_label = data['labelsTrain']
    test_data = data['wordsTest'].transpose()
    test_label = data['labelsTest']
    word_embeddings = data['embeddings']
    voc = data['vocabulary']
    voc = [v[0][0] for v in voc]
    test1 = data['test1'].transpose()
    test2 = data['test2'].transpose()

    if is_to_dense:
        train_data = sparse2dense(train_data)
        test_data = sparse2dense(test_data)
        test1 = sparse2dense(test1)
        test2 = sparse2dense(test2)

    return train_data, train_label, test_data, test_label, word_embeddings, voc, test1, test2


# load noisy target corpus
def load_noisy_target(aug_data_file, is_to_dense=True):
    data = sio.loadmat(aug_data_file)
    train_data = data['wordsTrain'].transpose()
    train_label = data['labelsTrain']
    test_data = data['wordsTest'].transpose()
    test_label = data['labelsTest']

    if is_to_dense:
        train_data = sparse2dense(train_data)
        test_data = sparse2dense(test_data)

    return train_data, train_label, test_data, test_label


# load source to target corpus
def load_combined_data(mat_file_name, is_to_dense=True):
    data = sio.loadmat(mat_file_name)
    train_data = data['wordsTrain_source'].transpose()
    train_label = data['labelsTrain_source']
    test_data = data['wordsTest_source'].transpose()
    test_label = data['labelsTest_source']
    word_embeddings = data['embeddings']
    voc = data['vocabulary']
    voc = [v[0][0] for v in voc]
    test1 = data['test1_source'].transpose()
    test2 = data['test2_source'].transpose()

    train_data_target = data['wordsTrain_target'].transpose()
    train_label_target = data['labelsTrain_target']
    test_data_target = data['wordsTest_target'].transpose()
    test_label_target = data['labelsTest_target']

    if is_to_dense:
        train_data = sparse2dense(train_data)
        test_data = sparse2dense(test_data)
        test1 = sparse2dense(test1)
        test2 = sparse2dense(test2)
        train_data_target = sparse2dense(train_data_target)
        test_data_target = sparse2dense(test_data_target)

    return train_data, train_label, test_data, test_label, word_embeddings, voc, test1, test2,\
        train_data_target, train_label_target, test_data_target, test_label_target


# for 20News as the source, others as targets
def load_union_data(mat_file_name):
    data = sio.loadmat(mat_file_name)
    # source data
    train_data = sparse2dense(data['wordsTrain_source'].transpose())
    train_label = data['labelsTrain_source']
    test_data = sparse2dense(data['wordsTest_source'].transpose())
    test_label = data['labelsTest_source']
    word_embeddings = data['embeddings']
    voc = data['vocabulary']
    voc = [v[0][0] for v in voc]
    test1 = sparse2dense(data['test1_source'].transpose())
    test2 = sparse2dense(data['test2_source'].transpose())

    # target data
    train_data_target_list = []
    train_label_target_list = []
    test_data_target_list = []
    test_label_target_list = []
    for i in range(1,5):
        train_data_target_list.append(sparse2dense(data['wordsTrain_target%s' % i].transpose()))
        train_label_target_list.append(sparse2dense(data['labelsTrain_target%s' % i].transpose()))
        test_data_target_list.append(sparse2dense(data['wordsTest_target%s' % i].transpose()))
        test_label_target_list.append(sparse2dense(data['labelsTest_target%s' % i].transpose()))

    return train_data, train_label, test_data, test_label, word_embeddings, voc, test1, test2, \
           train_data_target_list, train_label_target_list, test_data_target_list, test_label_target_list