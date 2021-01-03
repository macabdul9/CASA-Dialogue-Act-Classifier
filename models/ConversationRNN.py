import torch.nn as nn

class ConversationRNN(nn.Module):
    
    def __init__(self, input_size=1, hidden_size=768, bidirectional=True, num_layers=1):
        super(ConversationRNN, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional,
            batch_first=True
        )
        
    
    def forward(self, input_, hx=None):
        
        """
            input_.shape = [batch, input_size] # input_size was chosen in attention module
            hx.shape = [2, batch_size, hidden_size]
        """
        
        _, hidden = self.rnn(input=input_, hx=hx)
        
        return hidden