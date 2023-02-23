import time
import math
import torch.nn as nn
import torch.optim as optim
from simple_chatbot.training.train import train
from simple_chatbot.training.eval import evaluate
from simple_chatbot.modeling.seq2seq import Seq2Seq
from simple_chatbot.modeling.encoder import Encoder
from simple_chatbot.modeling.decoder import Decoder
from simple_chatbot.preprocessing.tokenization import load_dataset
from simple_chatbot.modeling.params import *
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch
import spacy


spacy_en = spacy.load('en_core_web_sm')
N_EPOCHS = 10
CLIP = 1


train_iterator, valid_iterator, test_iterator, src, trg = load_dataset("WikiQACorpus/", batch_size, device)
encoder = Encoder(len(src.vocab), embedding_size, hidden_size, num_layers, dropout).to(device)
decoder = Decoder(len(trg.vocab), embedding_size, hidden_size, num_layers, dropout).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)
PATH = "bigdata-chatbot-model.pt"
model.load_state_dict(torch.load(PATH))

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = src
TRG = trg

model.eval()

def generate_response(input_text):
    input_tokens = tokenize_en(input_text)

    input_indices = [SRC.vocab.stoi[token] for token in input_tokens]

    input_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(device)

    encoder_hidden, encoder_cell = model.encoder(input_tensor)

    decoder_input = torch.LongTensor([TRG.vocab.stoi['<pad>']]).to(device)

    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    output_tokens = []

    for i in range(20):
        output, decoder_hidden, decoder_cell = model.decoder(decoder_input, decoder_hidden, decoder_cell)

        top1 = output.argmax(1)
        output_tokens.append(top1.item())
        if top1.item() == TRG.vocab.stoi['<pad>']:
            break

        decoder_input = top1.unsqueeze(0)
    output_text = [TRG.vocab.itos[token] for token in output_tokens]
    return ' '.join(output_text)

while True:
    input_text = input('You: ')
    output_text = generate_response(input_text)
    print('Bot:', output_text)
