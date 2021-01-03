# CASA-Dialogue-Act-Classifier
PyTorch implementation of the paper [**Dialogue Act Classification with Context-Aware Self-Attention**](https://arxiv.org/abs/1904.02594) for dialogue act classification with a generic dataset class and PyTorch-Lightning trainer. This implementation has following changes compare to the actual paper
- In this implementation Contextualized Embedding (ie: BERT, RoBERta, etc ) (freezed hence not trainable) is used while paper uses combination of GloVe and ELMo
- This implementation has simple classifier but paper has CRF Classifier


To Run this on any dialogue act dataset:

- Download the Dataset and change the directory in config file
- Change the fields in dataset class
- Run main.py
  
Note: Scripts was generated form running notebook so if there is any problem in running this please feel free to create an issue. 