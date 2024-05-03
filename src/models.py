import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transformer(nn.Module):

    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        n_head: int,
        d_hid: int,
        n_layers: int,
        dropout: float = 0.5,
    ):
        """
        Arguments:
            n_tokens: int, the size of the vocabulary - number of unique tokens in the input data
            d_model: int, the number of expected features in the input
            n_head: int, the number of heads in the multiheadattention models
            d_hid: int, the dimension of the feedforward network model
            n_layers: int, the number of sub-encoder-layers in the encoder
            dropout: float, the dropout value
        """
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, n_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, n_tokens]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(
                device
            )
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LSTM(nn.Module):
    """
    Class for the LSTM model.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
        
    # def parameters(self):
    #     """
    #     Returns the parameters of the LSTM model.
    #     """
    #     return self.lstm.parameters()

    def forward(self, x):
        """
        Forward pass of the LSTM model.
        """
        h_0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), self.hidden_size)
        c_0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), self.hidden_size)
        
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        return out
        


        
