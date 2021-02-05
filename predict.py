from config import config
import numpy as np
import torch

from Trainer import LightningModel
from transformers import AutoTokenizer


def run_test(checkpoint_path, config, device='cpu'):
    # my_device = torch.device('cuda')
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

        def encode(text):
            input_encoding = self.tokenizer.encode_plus(
                text=text,
                truncation=True,
                max_length=self.config['max_len'],
                return_tensors='pt',
                return_attention_mask=True,
                padding='max_length',
            )
            seq_len = len(self.tokenizer.tokenize(text))

            return input_encoding['input_ids'].squeeze(), input_encoding['attention_mask'].squeeze(), seq_len

        # there is probably a more optimized way to map the inputs
        inputs['input_ids'], inputs['attention_mask'], inputs['seq_len'] = np.array(list(zip(*map(encode, data))))
        return inputs

    def predict(self, df):
        input = self.dataloader(df)
        with torch.no_grad():
            # model prediction labels
            outputs = self.model.model(input).argmax(dim=-1).tolist()
        return outputs
        # ort_outs = self.ort_session.run(None, self.dataloader(df))
        # return ort_outs


if __name__ == '__main__':
    ckpt_path = 'checkpoints/epoch=28-val_accuracy=0.746056.ckpt'
    run_test(ckpt_path, config, device='cpu')

    clf = DialogClassifier(checkpoint_path=ckpt_path, config=config)
    testing_data = ['Uh-huh.', 'Well, I think its a pretty good idea.', 'Okay.']
    print(clf.predict(testing_data))


