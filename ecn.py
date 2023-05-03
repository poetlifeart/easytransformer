import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        """
        This class is used to compute the sinusoid encoding of the positions of the words in a sequence.

        args:
            d_model (int): the dimension of the positional encoding which is equal to the 
                dimension of the word embedding
            max_len (int): the maximum length of the sequence
        """
        super().__init__()

        # create a tensor with the dimension of the positional encoding, max_len is the maximum length of the sequence
        # and d_model is the dimension of the word embedding.
        encode=torch.zeros((max_len, d_model), dtype=torch.float32)

        # set up the a range of possible values for the positions of the words in the sequence
        pos = torch.arange(0, max_len, dtype=torch.float32)
        # setup the argument of the sinusoid function for the even and odd indices of the positions of the words
        evenumbers = torch.arange(0, d_model, step = 2, dtype = torch.float32)
        # note the argument is always given by an even number, even indecies map to sin and odd indices map to cos
        arg = torch.outer(pos, (10000 **( -evenumbers / d_model)))

        # compute the positional encoding
        encode[:, 0::2] = torch.sin(arg)
        encode[:, 1::2] = torch.cos(arg)

        # we use register_buffer to store our positional encoding as a buffer so that it is not updated during 
        # training furthermore, this allows for the positional encoding to be moved to the GPU, if available with 
        # the model parameters
        self.register_buffer('encoding', encode)
        
    def forward(self, x):
        """
        The forward method accepts a batch of sequences as input and returns the positional encoding for the sequence

        args:
            x (torch.Tensor): is an arbitrary dimensional tensor with its last dimension being the sequence length,
            we use the sequence length to cutoff the positional encoding to the correct length
        """

        seq_len = x.shape[-1]
        return self.encoding[:seq_len, :]

# Testing the PositionalEncoding class
# condition to run the code below
if __name__ == '__main__':

    # The constrcutor accepts the dimension of the feature vector of the word and 
    # the maximum length of the sequence as input
    position = PositionalEncoding (d_model = 8, max_len = 40)

    # The input to the positional encoding is a 2D tensor with 
    # we are testing with batch size of 1 and sequence length of 20
    x = torch.arange(0, 20, dtype = torch.float32)
    x = x.unsqueeze(dim = 0)
    print(position.forward(x).shape)