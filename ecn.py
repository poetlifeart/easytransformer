import torch
import torch.nn as nn
 

#this class is used to compute the sinusoid encoding 
class PositionalEncoding(nn.Module):

    # The constructor accepts the dimension of the positional encoding and the maximum length of the sequence as input
    # The dimension of the positional encoding is equal to the dimension of the word embedding; i.e the dimension of feature vector of the word
    def __init__(self, d_model, max_len):
        super().__init__()

        # we use register_buffer to store the positional encoding as a buffer so that it is not updated during training
        # furthermore, this allows for the positional encoding to be moved to the GPU, if available with the model parameters
        self.register_buffer('encoding', torch.zeros((max_len, d_model), dtype=torch.float32))


        # set up the a range of possible values for the positions of the words in the sequence
        # setup the argument of the sinusoid function for the even and odd indices of the positins of the words
        # note the argument is always given by an even number, even indecies map to sin and odd indices map to cos
        pos = torch.arange(0, max_len, dtype=torch.float32)
        evenumbers = torch.arange(0, d_model, step=2, dtype=torch.float32)
        arg = torch.outer(pos, (10000 **( -evenumbers / d_model)))

        
        # compute the positional encoding
        self.encoding[:, 0::2] = torch.sin(arg)
        self.encoding[:, 1::2] = torch.cos(arg)
        
    # The forward method accepts a batch of sequences as input and returns the positional encoding for the sequence
    def forward(self, x):
        
        # x is a 2D tensor with dimensions batch size and sequence length
        # we use the sequence length to cutoff the positional encoding to the correct length
        batch_size, seq_len = x.shape
        return self.encoding[:seq_len, :]


# Testing the PositionalEncoding class
# condition to run the code below
if __name__ == "__main__":

    # The constrcutor accepts the dimension of the feature vector of the word and the maximum length of the sequence as input
    position = PositionalEncoding (d_model=8, max_len=40)

    # The input to the positional encoding is a 2D tensor with 
    # we are testing with batch size of 1 and sequence length of 20
    x = torch.arange(0, 20, dtype=torch.float32)
    x = x.unsqueeze(dim=0)
    print(position.forward(x).shape)



