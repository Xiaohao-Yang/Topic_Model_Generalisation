# Topic Model Generalisation
Neural topic model (NTM) generalisation in terms of document representation. 

We propose a Generalisation Regularisation (Greg) module to improve NTMs' generalisation capability in terms of document representation. As a result, an NTM trained on a source corpus still yields good document representation for unseen documents from other corpora.

See details of our [Paper]().

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
We use '20News', 'R8', 'Webs', 'TMN' and 'DBpedia' (a random subset), for our experiments. The pre-processed datasets are available for download at: https://drive.google.com/drive/folders/1aNpsTkd95yybj2cXAuwmgshFwBHgv1eF?usp=drive_link

We store our pre-processed datasets in .mat files, which can be loaded as dictionaries using scipy.io.loadmat(). The datasets/dictionaries have the following common attributes/keys:
* wordsTrain, labelsTrain: bag-of-words (BOW) of training documents, and their labels. 
* wordsTest, labelsTest: BOW of testing documents, and their labels.
* vocabulary, embeddings: vocabularies of the corpus, and their word embeddings from 'glove-wiki-gigaword-50'.
* test1, test2: the first and second fold of the test BOWs (for computing document completion perplexity). 

For source-to-target tasks, the source and target data have an extra suffix (e.g. 'wordsTrain_source' and 'wordsTrain_target').

For source-to-noisy tasks, the noisy target is stored in a separate 'data_aug.mat' file.

# Run topic models with Greg
To run original topic models:
```python
python main.py --model NVDM --dataset combined_20News_RestAll --n_topic 50
```

To run topic models with Greg:
```python
python main.py --model NVDM --dataset combined_20News_RestAll --n_topic 50 --use_Greg
```

# Results
We evaluate the document representation quality at every evaluation step (e.g. --eval_step=5). The evaluation is done by document classification and clustering for both source and target documents.

A running example without Greg at epoch 50:
```python
############################################
Evaluation at: 
NVDM_dataset:combined_20News_RestAll_K50_RS1_epochs:50_LR0.0003_reg:False_regW300.0_augRate:0.5_aug:DA2

doc classification acc (original corpus):  0.3967073818374934
doc classification acc (R8):  0.7035830618892508
doc classification acc (DBpedia):  0.2738184546136534
doc classification acc (TMN):  0.3679447852760736
doc classification acc (Webs):  0.362753036437247
############################################
doc clustering TP, TN (original corpus):  0.16330323951141795 0.11003465044205976
doc clustering TP, TN (R8):  0.6319218241042345 0.1279169527204744
doc clustering TP, TN (DBpedia):  0.1827956989247312 0.07871181078741796
doc clustering TP, TN (TMN):  0.3214723926380368 0.034599746750148555
doc clustering TP, TN (Webs):  0.2813765182186235 0.06003729503945232
############################################
source document completion ppl:  15523.4
############################################
```

A running example with Greg at epoch 50:
```python
############################################
Evaluation at: 
NVDM_dataset:combined_20News_RestAll_K50_RS1_epochs:50_LR0.0003_reg:True_regW300.0_augRate:0.5_aug:DA2

doc classification acc (original corpus):  0.4159585767392459
doc classification acc (R8):  0.7817589576547231
doc classification acc (DBpedia):  0.49087271817954486
doc classification acc (TMN):  0.5625766871165644
doc classification acc (Webs):  0.5817813765182186
############################################
doc clustering TP, TN (original corpus):  0.16648964418481146 0.11442718376169024
doc clustering TP, TN (R8):  0.6436482084690553 0.15272100820116064
doc clustering TP, TN (DBpedia):  0.24081020255063765 0.14740831169213556
doc clustering TP, TN (TMN):  0.4170245398773006 0.10636887676325055
doc clustering TP, TN (Webs):  0.33765182186234816 0.11487270835789015
############################################
source document completion ppl:  15433.4
############################################
```

Here is one of our results (Table 4), for 5 runs of 20News as the source, and the rest as the targets, where the number of topics for models is set as 50. 
<p align="center">
  <img src="results.png" alt="results" width="1000"/>
</p>
Overall, Greg brings significant (Table 9-11) improvements to the original models in most cases regarding neural topical generalisation. See more details in our [Paper]().

# References
Our code is based on the following implementations:

* For NVDM: [Code](https://github.com/visionshao/NVDM).
* For PLDA: [Code](https://github.com/estebandito22/PyTorchAVITM).
```python
@MISC {Carrow2018,
    author       = "Stephen Carrow",
    title        = "PyTorchAVITM: Open Source AVITM Implementation in PyTorch",
    howpublished = "Github",
    month        = "dec",
    year         = "2018"
}
```
* For SCHOLAR and CLNTM: [Code](https://github.com/nguyentthong/CLNTM).
```python
@inproceedings{
nguyen2021contrastive,
title={Contrastive Learning for Neural Topic Model},
author={Thong Thanh Nguyen and Anh Tuan Luu},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=NEgqO9yB7e}
}
```
* For Pytorch Sinkhorn distance: [Code](https://github.com/ethanhezhao/Tensorflow_Pytorch_Sinkhorn_OT)


# Citation 

