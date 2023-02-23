import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[0]
        target_length = target.shape[1]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(target_length, batch_size, target_vocab_size)
        hidden, cell = self.encoder(input)

        decoder_input = target[:, 0]
        for t in range(1, target_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1

        return outputs