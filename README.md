# CASA-Dialogue-Act-Classifier
PyTorch implementation of the paper [**Dialogue Act Classification with Context-Aware Self-Attention**](https://arxiv.org/abs/1904.02594) for dialogue act classification with a generic dataset class and PyTorch-Lightning trainer. This implementation has following differences compare to the actual paper
- In this implementation contextualized embedding (ie: BERT, RoBERta, etc ) (freezed hence not trainable) is used while paper uses combination of GloVe and ELMo.
- This implementation has simple softmax classifier but paper has CRF classifier.


## To train this on switchboard dialogue act dataset:
  1. Download the cleaned [Switchboard](https://www.dropbox.com/s/bpkt44sijmhfbxq/switchboard.zip) data inside the `data/` and unzip it.
Navigate to `data/` using: `cd data/`
  2. Download the [Switchboard](https://www.dropbox.com/s/bpkt44sijmhfbxq/switchboard.zip) dataset: `wget https://www.dropbox.com/s/bpkt44sijmhfbxq/switchboard.zip?`
  3. Unzip the dataset: `unzip switchboard.zip`
  4. Navigate to the main dir: `cd ..`
  5. Install the dependencies in a separate python environment.
  6. [Optional] Change the project_name and run_name in the logger or disable the wandb logger if you don't want to use it by commenting the logger code (`line 15-20 in main.py`) and don't pass it to Lightning trainer (`line 32 in main.py`), and then comment the logging code in `Trainer.py (line 70 and 95)`.  By default Lightning will log to tensorboard logger.
  7. [Optional] Change the parameters (`batch_size, lr, epochs etc`) in `config.py`.
  8. Run main.py using `python main.py`
  9. Model will be trained and best checkpoint will be saved. 
  

## To train this on any dialogue act dataset 

1. Paste your data into `data/`, your dataset should have following structure
    - dataset_name
      - dataset_name_train.csv
      - dataset_name_valid.csv
      - dataset_name_test.csv
2. [Optional] If you don't have separate test and validation data, copy the test/valid and rename it as valid/test, this both validation and test data will be same. 
3. Update the num_classes param in `config.py line 18` according to your dataset.
4. Follow from steps 5 of the switchboard.
  

**Note**: Feel free to create to an issue if you find any problem. Also you're welcome to create PR if you want to add something. Here is the list of components one can add:
- Hyperparameter Search
- More dialogue act classification models which are not open-sourced. 


  
## *References*
**[1]:** Raheja, V., & Tetreault, J. (2019). Dialogue Act Classification with Context-Aware Self-Attention. ArXiv, abs/1904.02594.

**[2]:** Lin, Z., Feng, M., Santos, C.D., Yu, M., Xiang, B., Zhou, B., & Bengio, Y. (2017). A Structured Self-attentive Sentence Embedding. ArXiv, abs/1703.03130.

**[3]:** Switchboard Dialogue Act corpus: http://compprag.christopherpotts.net/swda.html