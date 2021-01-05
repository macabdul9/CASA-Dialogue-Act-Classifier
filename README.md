# CASA-Dialogue-Act-Classifier
PyTorch implementation of the paper [**Dialogue Act Classification with Context-Aware Self-Attention**](https://arxiv.org/abs/1904.02594) for dialogue act classification with a generic dataset class and PyTorch-Lightning trainer. This implementation has following differences compare to the actual paper
- In this implementation contextualized embedding (ie: BERT, RoBERta, etc ) (freezed hence not trainable) is used while paper uses combination of GloVe and ELMo.
- This implementation has simple softmax classifier but paper has CRF classifier.


## To run this on switchboard dialogue act dataset:
- Set the dataset path (absolute path) as per your system configuration, in the `config.py (line 6)`.   
- Install the dependencies in a separate python environment and activate the environment.
- [Optional] Disable the wandb logger if you don't want to use it by commenting the logger code (`line 15-20 in main.py`) and don't pass it to Lightning trainer (`line 32 in main.py`), and then comment then comment the logging code in Trainer.py (`line 70 and 95`).  By default Lightning will log to tensorboard logger.
- Run main.py using `python main.py`
- Model will be trained and best checkpoint will be saved. 
  

## To on this on any dialogue act dataset 

- Paste you data into `data dir`, your dataset should have following structure
    - dataset_name
      - dataset_name_train.csv
      - dataset_name_valid.csv
      - dataset_name_test.csv
- [Optional] If you don't have separate test and validation data, copy the test/valid and rename it as valid/test, this both validation and test data will be same. 
- Update the num_classes param in `config.py line 18` according to your dataset.
- Change the dataset path (absolute) and data name in `config.py (line 6 and 7 respectively)` 
- [Optional] Disable the logger as describe in earlier section
- Run main.py using `python main.py`
- Model will be trained and best checkpoint will be saved.

**Note**: Feel free to create to an issue if you find any problem. Also you're welcome to create PR if you want to add something. Here is the list of components one can add:
- Hyperparameter Search
- More dialogue act classification models which are not open-sourced. 


  
## *References*
**[1]:** Raheja, V., & Tetreault, J. (2019). Dialogue Act Classification with Context-Aware Self-Attention. ArXiv, abs/1904.02594.

**[2]:** Lin, Z., Feng, M., Santos, C.D., Yu, M., Xiang, B., Zhou, B., & Bengio, Y. (2017). A Structured Self-attentive Sentence Embedding. ArXiv, abs/1703.03130.

**[3]:** Switchboard Dialogue Act corpus: http://compprag.christopherpotts.net/swda.html