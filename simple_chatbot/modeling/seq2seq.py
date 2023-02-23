import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    """Seq2Seq model.
        we first initialize the encoder and decoder, as well as the device to be used (e.g. CPU or GPU). We then define the /
        forward method, which takes in the input text, input_lengths, output text, and teacher_forcing_ratio.

        We first initialize the output tensor with zeros, the hidden and cell states using the encoder, and the input text with the /
        first token of the output text. We then loop through the remaining tokens of the output text, using the Decoder to generate /
        the next token given the current input text, hidden state, and cell state. We then update the output tensor with the generated /
        token, and either use the ground truth token or the predicted token as input to the Decoder for the next time step, based on /
        the teacher_forcing_ratio.
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1
        
        return outputs