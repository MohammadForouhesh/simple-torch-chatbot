import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    we are defining the Encoder and Decoder classes using PyTorch's nn.Module class. The Encoder takes in the input text, /
    embeds it, runs it through an LSTM layer, and returns the final hidden and cell states. The Decoder takes in the representation, /
    embeds it, runs it through an LSTM layer, and produces the output text.
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        
        return hidden, cell
