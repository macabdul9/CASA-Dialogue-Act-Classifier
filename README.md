# CASA-Dialogue-Act-Classifier
PyTorch implementation of the paper [**Dialogue Act Classification with Context-Aware Self-Attention**](https://arxiv.org/abs/1904.02594) for dialogue act classification with a generic dataset class and PyTorch-Lightning trainer. This implementation has following differences compare to the actual paper
- In this implementation Contextualized Embedding (ie: BERT, RoBERta, etc ) (freezed hence not trainable) is used while paper uses combination of GloVe and ELMo.
- This implementation has simple softmax classifier but paper has CRF classifier.


## To Run this on any dialogue act dataset:
- Install the dependencies in a separate python environment.
- Download the dataset and change the directory for th same in config file. 
- Change the fields in dataset class
- Run main.py
  
Note: Scripts was generated form running notebook so if there is any problem in running this please feel free to create an issue. 