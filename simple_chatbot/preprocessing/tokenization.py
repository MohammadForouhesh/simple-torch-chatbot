from typing import Tuple
import torchtext
from torchtext.legacy.data import Field, TabularDataset, BucketIterator


def load_dataset(path: str, batch_size: int, device: str) -> Tuple[BucketIterator, BucketIterator, BucketIterator, Field]:
    """
    we are using the torchtext library to define the fields for the input and output text, load the WikiQA dataset, /
    uild the vocabulary for the input text, and set up the iterators for the training and validation sets. /
    Note that we are tokenizing the text using the spacy tokenizer, and limiting the vocabulary size to 10,000 words.
    Args:
        path (str): path to the file containing the dataset
        batch_size (int): The batch size for the dataset
        device (str): The device to use for the dataset

    Returns:
        _type_: Tuple[BucketIterator, BucketIterator, BucketIterator, Field]
    """
    SRC = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
    TRG = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)

    # Load the WikiQA dataset
    train_data, valid_data, test_data = TabularDataset.splits(
        path=path, train='WikiQA-train.tsv', validation='WikiQA-dev.tsv', test='WikiQA-test.tsv',
        format='tsv', skip_header=True,
        fields=[('questionId', None), ('question', SRC), ('id', None), ('title', None), ('sentence_id', None), ('answer', TRG), ('label', None)])


    SRC.build_vocab(train_data, max_size=len(train_data))
    TRG.build_vocab(train_data, max_size=len(train_data))

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size, device=device)

    return train_iterator, valid_iterator, test_iterator, SRC, TRG

