import os
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

# for converting model to onnx:
import onnx
from config import config
import torch
from Trainer import LightningModel


def convert_onnx(model_path, config):
    # my_device = torch.device('cuda')
    my_device = torch.device('cpu')

    model = LightningModel(config=config)
    model = model.to(my_device)
    model.load_state_dict(torch.load(model_path, map_location=my_device)['state_dict'])

    example_loader = model.test_dataloader()
    batch = next(iter(example_loader))
    print(batch['text'])
    print(batch['label'])

    example_input = {'input_ids': batch['input_ids'].to(my_device),
                     'attention_mask': batch['attention_mask'].to(my_device),
                     'seq_len': batch['seq_len'].to(my_device)}

    with torch.no_grad():
        # model prediction labels
        outputs = model.model(example_input).argmax(dim=-1).tolist()
    print(outputs)

    outname = os.path.splitext(model_path)[0] + '.onnx'
    # issues tracing some operators in model
    torch.onnx.export(
        model,
        args=(example_input,),
        f=outname,
        input_names=['input_ids', 'attention_mask', 'seq_len'],
        output_names=['label'],
        do_constant_folding=True,
        export_params=True,
        opset_version=11,
    )

    model = onnx.load(outname)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))

    # check if predictions match with exported model
    ort_session = ort.InferenceSession(outname)
    example_input = dict((k, v.detach().cpu().numpy()) for k, v in example_input.items())

    ort_outs = ort_session.run(None, example_input).tolist()
    print(ort_outs)

    assert ort_outs == outputs
    print("ONNX Model exported to {0}".format(outname))
    return


class DialogClassifier:

    def __init__(self, model_path, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.ort_session = ort.InferenceSession(model_path)

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
        ort_outs = self.ort_session.run(None, self.dataloader(df))
        return ort_outs


if __name__ == '__main__':
    convert_onnx(model_path='checkpoints/epoch=28-val_accuracy=0.746056.ckpt', config=config)

    clf = DialogClassifier(model_path='checkpoints/epoch=11-val_accuracy=0.746056.onnx', config=config)
    testing_data = ['Uh-huh.', 'Well, I think its a pretty good idea.', 'Okay.']
    print(clf.predict(testing_data))

