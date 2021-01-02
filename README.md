# CASA-Dialogue-Act-Classifier
PyTorch implementation of the paper `Dialogue Act Classification with Context-Aware Self-Attention` for dialogue act classification with a generic dataset class and PyTorch-Lightning trainer. This implement has following differences than the actual paper
- In this implementation Contextualized Embedding (ie: BERT, RoBERta, etc ) (freezed hence not trainable) is used while paper uses combincation of GloVe and ELMo
- This implementation has simple classifier but paper has CRF Classifier
