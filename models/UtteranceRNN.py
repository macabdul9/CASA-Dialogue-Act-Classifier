
import  torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class UtteranceRNN(nn.Module):
    
    def __init__(self, model_name="roberta-base", hidden_size=768, bidirectional=True, num_layers=1):
        super(UtteranceRNN, self).__init__()
        
        
        # embedding layer is replaced by pretrained roberta's embedding
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        # freeze the model parameters
        for param in self.base.parameters():
            param.requires_grad = False
        
        #self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.rnn = nn.RNN(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional,
            batch_first=True
        )
    
    def forward(self, input_ids, attention_mask, seq_len):
        """
            x.shape = [batch_size, seq_len]
        """
        
    
        hidden_states, _ = self.base(input_ids, attention_mask) # hidden_states.shape = [batch, max_len, hidden_size]
        
        # padding and packing 
        #packed_hidden_states = nn.utils.rnn.pack_padded_sequence(hidden_states, seq_len, batch_first=True, enforce_sorted=False)   
        
        #packed_outputs, _ = self.rnn(packed_hidden_states)
        
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
        
        #outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        outputs,_ = self.rnn(hidden_states)
                
        return outputs