import torch
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transformer(nn.Module):
    def __init__(self, input_size, n_head, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.input_size = input_size
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.transformer = nn.Transformer(d_model=input_size,
                                         nhead=n_head,
                                         num_encoder_layers=num_encoder_layers,
                                         num_decoder_layers=num_decoder_layers,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout)
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass of the Transformer model.
        """
        x = self.transformer(x, x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    


class LSTM(nn.Module):
    """
    Class for the LSTM model.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.1)
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.label = nn.Linear(128, output_size)
        self.fc = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the LSTM model.
        """
        h_0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n = self.dropout(h_n)
        x = self.fc(h_n[-1])
        x = self.relu(x)
        x = self.label(x)
        return x        
        

        


        
