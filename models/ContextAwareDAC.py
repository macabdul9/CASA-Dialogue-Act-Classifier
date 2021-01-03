import torch.nn as nn
import torch
from UtteranceRNN import UtteranceRNN
from ConversationRNN import ConversationRNN
from ContextAwareAttention import ContextAwareAttention



class ContextAwareDAC(nn.Module):
    
    def __init__(self, model_name="roberta-base", hidden_size=768, num_classes=18):
        
        super(ContextAwareDAC, self).__init__()
        
        self.in_features = 2*hidden_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # utterance encoder model
        self.utterance_rnn = UtteranceRNN(model_name=model_name, hidden_size=hidden_size)
        
        # context aware self attention module
        self.context_aware_attention = ContextAwareAttention(hidden_size=2*hidden_size, output_size=hidden_size, seq_len=128)
        
        # conversaton level rnn
        self.conversation_rnn = ConversationRNN(input_size=1, hidden_size=hidden_size)
        
        # classifier on top of feature extractor
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=self.in_features, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        ])
        
        # initial hidden_states
        self.hx = torch.randn((2, 1, hidden_size), device=self.device)
        
    
    def forward(self, batch):
        """
            x.shape = [batch, seq_len, hidden_size]
        """
        
        
        outputs = self.utterance_rnn(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], seq_len=batch['seq_len'].tolist())
        
        batch = batch['input_ids'].shape[0]
        
        # create an empty feature vector 
        features = torch.empty((0, self.in_features), device=self.device)
        
        # hidden
        hx = self.hx
        
    
        for i, x in enumerate(outputs):
            
            x = x.unsqueeze(0)
            
            # get sentence representation as 2d-matrix and project it linearly
            m = self.context_aware_attention(hidden_states=x, h_forward=hx[0].detach())
            
            # apply rnn on linearly projected vector
            hx = self.conversation_rnn(input_=m, hx=hx.detach())
            
            # concat current utterance's last hidden state to the features vector
            features = torch.cat((features, hx.view(1, -1)), dim=0)
            
        
        self.hx = hx.detach()
        
        logits = self.classifier(features)
        
        return logits
          