import torch

hidden_size = 2
embedding_size = 2
num_layers = 1
dropout = 0.5
learning_rate = 0.001
batch_size = 2
num_epochs = 10
clip = 1
teacher_forcing_ratio = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
