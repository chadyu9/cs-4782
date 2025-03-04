import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gensim


def get_word_embeddings(model, sequence):
    """
    Inputs:
    model: Word2Vec (TODO 1) or KeyedVectors (TODO 4) model.
    sequence: A single review (sequence of words).

    Outputs:
    word_embeddings: The word embeddings for each word in the input sequence. If a word in the
                     sequence is not included in the word2vec model's vocabulary, that word
                     should be ignored and should not be added to the output word_embeddings.
    """
    word_embeddings = []

    ## TODO 1: implement function for Word2Vec model
    ## TODO 4: add support for KeyedVectors model
    #################

    return word_embeddings


def get_reviews_embeddings(model, data):
    """
    Inputs:
    model: word2vec model(any).
    data: Cleaned dataframe containing n reviews .
          This dataframe has two coloumns: 'text' (the review) and 'label' (whether the review is positive or negative).

    Outputs:
    reviews_embeddings: A list of length n, containing the word embeddings for each
                        review in the input data generated using the input Word2Vec model, i.e.
                        review_embeddings[i] are the word embeddings for the (i + 1)th review
                        in the dataframe.
    """
    reviews_embeddings = []
    ## TODO 2: implement function
    #################

    return reviews_embeddings


def max_pool(embeddings, d=300):
    """
    Input:
    embeddings (l x d): The word embeddings of a single review.
    d: The dimension of the embedding.

    Output:
    pooled_embeddings (d): Max-pooled word embeddings.
    """
    max_pool_embeddings = []
    ## TODO 3: max pooling
    #################
    return max_pool_embeddings

class LSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, seq_length, num_classes=2):
        """
        Inputs:
        num_layers: Number of recurrent layers
        input_size: Number of features for input
        hidden_size: Number of features in hidden state
        seq_length: Length of sequences in a batch
        num_classes: Number of categories for labels

        Outputs: none
        """
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        ## TODO 5: define LSTM components
        #################

    def forward(self, x):
        """
        Inputs:
        x: input data

        Outputs:
        out: output of forward pass
        """
        ## TODO 5: implement the forward function based on the architecture described above
        #################

        return x


def reviews_processing(google_embeddings, length):
    """
    This function takes in a dataset of embeddings and a length to clip these sequences to

    Inputs:
    google_embeddings: a dataset of n reviews that have been embedded with
                       google's w2v with embedding size d
    length: the length you are making sure all sequences are capped at

    Output:
    embeddings (n x length x d): a numpy array representing the dataset where all
                                 reviews are clipped to the input length. If the sequence
                                 is shorter than the specified length, pad the sequence
                                 with zeros.
    """
    embeddings = []
    ## TODO 6: Implement reviews_processing to modify the embeddings
    ###########

    return embeddings

def val(model, val_loader, criterion, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.

    Outputs:
    Tuple of (validation loss, validation accuracy)
    """
    val_running_loss = 0
    num_correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader, 0):

            # TODO 7: write validation loop body
            #################
            pass

    return val_running_loss, (num_correct / total).item()

def train(model, train_loader, val_loader, criterion, epochs, optimizer, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.
    epochs: Number of epochs to train for.
    optimizer: The optimizer to use during training.

    Outputs:
    Tuple of (train_loss_arr, val_loss_arr, val_acc_arr)
    """
    train_loss_arr = []
    val_loss_arr = []
    val_acc_arr = []
    running_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):

            # TODO 7: write train loop body
            #################
            pass

        val_loss, val_acc = val(model, val_loader, criterion, device)
        train_loss_arr.append(running_loss)
        val_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)
        print(
            "epoch:",
            epoch + 1,
            "training loss:",
            running_loss,
            "val accuracy:",
            round(val_acc, 3),
        )

    print(running_loss)
    print("Training finished.")

    return train_loss_arr, val_loss_arr, val_acc_arr


# TODO 8: Understand this code.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        max_seq_length: Maximum length of sequences input into the transformer.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).reshape(max_seq_length, 1)
        div_term = torch.exp( 
              -1 * (torch.arange(0, d_model, 2).float()/d_model) * math.log(10000.0)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Adds the positional encoding to the model input x.
        """
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: The number of attention heads to use.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO 9.1: define layers W_q, W_k, W_v, and W_o
        # Hint: Recall that linear layers essentially perform matrix multiplication
        #       between the layer input and layer weights
        #################

    def split_heads(self, x):
        """
        Reshapes Q, K, V into multiple heads.
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)

    def compute_attention(self, Q, K, V):
        """
        Returns single-headed attention between Q, K, and V.
        """
        # TODO 9.2: compute attention using the attention equation provided above
        #################
        attention = None
        return attention

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x):
        # TODO: 9.3 implement forward pass
        #################
        return x



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        d_ff: Hidden dimension size for the feed-forward network.
        """
        super(FeedForward, self).__init__()

        # TODO 10: define the network
        #################

    def forward(self, x):
        # TODO 10: implement feed forward pass
        #################

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        d_ff: Hidden dimension size for the feed-forward network.
        p: Dropout probability.
        """
        super(EncoderLayer, self).__init__()

        # TODO 11: define the encoder layer
        #################

    def forward(self, x):

        ## TODO 11: implement the forward function based on the architecture described above
        #################

        return x


class Transformer(nn.Module):
    def __init__(
        self, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, p
    ):
        """
        Inputs:
        num_classes: Number of classes in the classification output.
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension size for the feed-forward network.
        max_seq_length: Maximum sequence length accepted by the transformer.
        p: Dropout probability.
        """
        super(Transformer, self).__init__()

        # TODO 12: define the transformer
        #################

    def forward(self, x):

        ## TODO 12: implement the forward pass
        #################

        return x


def process_batch(bert_model, data, criterion, device, val=False):
    """
    Inputs:
    data: The data in the batch to process.
    criterion: The loss function.
    val: True if processing a batch from the validation or test set.
         False if processing a batching from the training set.

    Outputs:
    Tuple of (outputs, losses)
        outputs: a dictionary containing the model outputs ('out') and predicted labels ('preds')
        metrics: a dictionary containing the model loss over the batch ('loss') and during validation (val = True),
                 the total number of examples in the batch ('batch_size') and the total number of examples whose
                 label the model predicted correctly ('num_correct')
    """

    outputs, metrics = dict(), dict()

    # TODO 13: process batch
    # Hint: For details on what information the data from the data loader contains
    #       check the __getitem__ function defined in the CustomClassDataset implemented
    #       at the beginning of Part 5
    # Hint: Make sure to send the data to the same device that the model is on.
    #################

    return outputs, metrics
