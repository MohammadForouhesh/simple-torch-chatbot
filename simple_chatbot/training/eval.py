import torch
from torch.nn import Module, functional
from torchtext.legacy.data import BucketIterator


def evaluate(model: Module, iterator: BucketIterator, criterion: functional) -> torch.Tensor:
    """define the evaluate function, which takes in the model, iterator, and criterion, and returns the average epoch loss. 
    We loop through the iterator (which contains batches of input and output text), pass the input text, input_lengths, 
    and output text through the model (with teacher forcing turned off), compute the loss, and accumulate the epoch loss.

    Args:
        model (Module): model to evaluate 
        iterator (BucketIterator): data to evaluate the model on
        criterion (functional): criterion to use for loss calculation
        device (str): device to use for evaluation (e.g. CPU or GPU)

    Returns:
        torch.Tensor: the loss value for the batch of evaluation examples
    """
    model.eval()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        with torch.no_grad():
            src, _ = batch.question
            trg, _ = batch.answer
            output = model(src, trg, 0)

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)