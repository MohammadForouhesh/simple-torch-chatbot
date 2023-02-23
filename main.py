
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
N_EPOCHS = 10
CLIP = 1

train_iterator, valid_iterator, test_iterator, src, trg = load_dataset("WikiQACorpus/", batch_size, device)
encoder = Encoder(len(src.vocab), embedding_size, hidden_size, num_layers, dropout).to(device)
decoder = Decoder(len(trg.vocab), embedding_size, hidden_size, num_layers, dropout).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_valid_loss = float('inf')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, train_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'bigdata-chatbot-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')