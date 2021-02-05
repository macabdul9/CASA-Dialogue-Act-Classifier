from config import config
import torch
from transformers import AutoTokenizer

from Trainer import LightningModel


def run_test(checkpoint_path, config, device='cpu'):
    my_device = torch.device(device)

    model = LightningModel(config=config)
    model = model.to(my_device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=my_device)['state_dict'])

    test_dataloader = model.test_dataloader()
    batch = next(iter(test_dataloader))
    print(batch['text'])
    print(f"Labels: {batch['label'].squeeze()}")

    example_input = {'input_ids': batch['input_ids'].to(my_device),
                     'attention_mask': batch['attention_mask'].to(my_device),
                     'seq_len': batch['seq_len'].to(my_device)}

    with torch.no_grad():
        # model prediction labels
        outputs = model.model(example_input).argmax(dim=-1).tolist()
    print(f"Predictions: {outputs}")


class DialogClassifier:

    def __init__(self, checkpoint_path, config, my_device='cpu'):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.model = LightningModel(config=config)
        self.model = self.model.to(my_device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=my_device)['state_dict'])

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


if __name__ == '__main__':
    ckpt_path = 'checkpoints/epoch=28-val_accuracy=0.746056.ckpt'
    # run_test(ckpt_path, config, device='cpu')

    clf = DialogClassifier(checkpoint_path=ckpt_path, config=config, my_device='cpu')
    testing_data = ['Uh-huh.', 'Well, I think its a pretty good idea.', 'Okay.']
    predictions = clf.predict(testing_data)
    for utterance, prediction in zip(testing_data, predictions):
        print(f"{utterance}\nPredicted speech act: {prediction}")
