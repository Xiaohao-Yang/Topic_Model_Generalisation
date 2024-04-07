# Topic Model Generalisation
Neural topic model (NTM) generalisation in terms of document representation. 

We propose a Generalisation Regularisation (Greg) term to improve NTMs' generalisation capability in terms of document representation. As a result, an NTM trained on a source corpus still yields good document representation for unseen documents from other corpora.

# Requirements
```python
torch: 2.2.1+cu121
torchmetrics: 1.3.2
numpy: 1.24.1
scipy: 1.12.0
scikit-learn: 1.4.1.post1
gensim: 4.3.2
pot: 0.9.3
tqdm: 4.66.2
```

# Datasets
We use `20News', `R8', `Webs', `TMN' and `DBpedia' (a random subset), for our experiments. The pre-processed datasets are available for download at: https://drive.google.com/drive/folders/1aNpsTkd95yybj2cXAuwmgshFwBHgv1eF?usp=drive_link

We store our pre-processed datasets in .mat files, which can be loaded as dictionaries using scipy.io.loadmat(). The datasets/dictionaries have the following common attributes/keys:
* wordsTrain, labelsTrain: bag-of-words (BOW) of training documents, and their labels. 
* wordsTest, labelsTest: BOW of testing documents, and their labels.
* vocabulary, embeddings: vocabularies of the corpus, and their word embeddings from `glove-wiki-gigaword-50'.
* test1, test2: the first and second fold of the test BOWs (for computing document completion perplexity). 

For source-to-target tasks, the source and target data have an extra suffix (e.g. `wordsTrain_source' and `wordsTrain_target').

For source-to-noisy tasks, the noisy target is stored in a separate `data_aug.mat' file.

# Run topic models with Greg
To run original topic models:
```python
python main.py --model NVDM --dataset combined_20News_RestAll --n_topic 50
```

To run topic models with Greg:

# Results

# Citation 

