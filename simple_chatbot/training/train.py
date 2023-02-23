import torch
from torch.nn import Module, functional
from torch.optim import Optimizer
from torchtext.legacy.data import BucketIterator


def train(model: Module, iterator: BucketIterator, optimizer: Optimizer, criterion: functional.cross_entropy, clip: float) -> torch.Tensor:
    """We first define the training function, which takes in the input and output tensors, as well as the model, 
    optimizer, criterion, and teacher_forcing_ratio. It performs the forward and backward passes, and updates 
    the model parameters.

    Args:
        model (Module): model to train
        iterator (BucketIterator): data to train the model on
        optimizer (Optimizer): optimizer to use for training the model parameters 
        criterion (functional): criterion to use for loss calculation
        clip (float): gradient clipping value to prevent exploding gradients 
        device (str): device to use for training (e.g. CPU or GPU)
        teacher_forcing_ratio (float): teacher forcing ratio to use for training the model 

    Returns:
        torch.Tensor: the loss value for the batch of training examples
    """
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src, _ = batch.question
        trg, _ = batch.answer
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
