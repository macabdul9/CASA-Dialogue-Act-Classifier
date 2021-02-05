import os
import pandas as pd
import sys
import torch
from transformers import AutoTokenizer

from config import config
from Trainer import LightningModel


class DialogClassifier:
    """
    Class to perform inference from a pre-saved checkpoint
    """

    def __init__(self, checkpoint_path, config, my_device='cpu'):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.model = LightningModel(config=config)
        self.model = self.model.to(my_device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=my_device)['state_dict'])

    def get_classes(self):
        return self.model.classes

    def dataloader(self, data):
        if not isinstance(data, list):
            data = list(data)

        inputs = dict()

        input_encoding = self.tokenizer.batch_encode_plus(
            data,
            truncation=True,
            max_length=self.config['max_len'],
            return_tensors='pt',
            return_attention_mask=True,
            padding='max_length',
        )
        seq_len = [len(self.tokenizer.tokenize(utt)) for utt in data]

        inputs['input_ids'] = input_encoding['input_ids'].squeeze()
        inputs['attention_mask'] = input_encoding['attention_mask'].squeeze()
        inputs['seq_len'] = torch.Tensor(seq_len)

        return inputs

    def predict(self, df):
        input = self.dataloader(df)
        with torch.no_grad():
            # model prediction labels
            outputs = self.model.model(input).argmax(dim=-1).tolist()
        return outputs


def main(argv):
    """
    Predict speech acts for the utterances in input file
    :param argv: Takes 1 argument. File with utterances to classify, one per line.
    :return: Prints file with utterances tagged with speech act
    """

    input_file = argv[0]
    ckpt_path = 'checkpoints/epoch=28-val_accuracy=0.746056.ckpt'  # Modify to use your checkpoint

    clf = DialogClassifier(checkpoint_path=ckpt_path, config=config, my_device='cpu')  # Choose 'cuda' if desired
    classes = clf.get_classes()
    inv_classes = {v: k for k, v in classes.items()}  # Invert classes dictionary

    with open(input_file, 'r') as fi:
        utterances = fi.read().splitlines()

    predictions = clf.predict(utterances)
    predicted_acts = [inv_classes[prediction] for prediction in predictions]

    results = pd.DataFrame(list(zip(predicted_acts, utterances)), columns=["DamslActTag", "Text"])
    filename = os.path.basename(input_file)
    results.to_csv(os.path.splitext(filename)[0] + ".out", index=False)

    print("-------------------------------------")
    print("Predicted Speech Act, Utterance")
    print("-------------------------------------")

    for utterance, prediction in zip(utterances, predicted_acts):
        print(f"{prediction}, {utterance}")


if __name__ == '__main__':
    main(sys.argv[1:])
