import math
import torch
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
   
    def __init__(self, d_tensor):
        """
        The constructor accepts the dimension of the key and value vectors as input.
        This class accepts queries, keys, and values as inputs and returns the output of the attention mechanism
        as well as the attention weights. 

        args:
            d_tensor (int): the dimension of the key and value feature vectors
        """
        super().__init__()

        # The square root of the dimension of the key feature vector is stored in the self.scale attribute
        self.scale = math.sqrt(d_tensor)

    def forward(self, queries, keys, values, mask=None):
        """
        The forward method accepts queries, keys, and values as inputs and returns the output of the attention mechanism
        Each tensor has the following dimensions: batch size, number of heads, number of timesteps, 
        and dimension of the feature vector

        args:
            queries (torch.Tensor): a 4D tensor with shape (batch size, number of heads, number of timesteps, 
            and dimension of the feature vector)
            keys (torch.Tensor): a 4D tensor with dimensions batch size, number of heads, number of timesteps, 
            and dimension of the feature vector
            values (torch.Tensor): a 4D tensor with dimensions batch size, number of heads, number of timesteps,
            and dimension of the feature vector
            mask (torch.Tensor): a mask that is used to make the attention mechanism causal with dimensions batch size,
            number of heads, number of timesteps, and number of timesteps. The mask is a tensor with binary values 
            where 0 will mask the attention weighte. If the size of either batsize or number of heads is 1, 
            the mask if broadcasted to the correct size. 
        """

        # by transposing the keys, we can compute the dot product between the queries and keys via matrix multiplication
        # The queries and keys are multiplied together and divided by the square root of the dimension of the key vector
        attention = (queries @ keys.transpose(2,3))/self.scale

        # The mask can be used to make the attention mechanism causal
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))

        # The softmax function is applied to the attention weights to obtain the output of the attention mechanism
        # The output of the attention mechanism is multiplied by the values to obtain 
        # the final output of the attention mechanism
        attention = torch.softmax(attention, dim=-1)
        values = attention @ values
        return values, attention

# Testing the ScaleDotProductAttention class
if __name__ == '__main__':

    # The queries, keys, and values are 4D tensors with the following dimensions: batch size, number of heads, 
    # number of timesteps, and dimension features
    batch_size = 2
    heads = 3
    timesteps = 8
    feats = 4
    keys = torch.rand(batch_size, heads, timesteps, feats)
    queries = torch.rand(batch_size, heads, timesteps, feats)
    values = torch.rand(batch_size, heads, timesteps, feats)

    # d_tensor is the dimension of the key and value feature vectors
    d_tensor = keys.shape[-1]
    test = ScaleDotProductAttention(d_tensor)
    values, attention = test(queries,keys,values)
    print(values.shape)