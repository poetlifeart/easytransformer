import math 
import torch
import torch.nn as nn

#This class accepts queries, keys, and values as inputs and returns the output of the attention mechanism
#as well as the attention weights. 

class ScaleDotProductAttention(nn.Module):

    # The constructor accepts the dimension of the key and value vectors as input.
    def __init__(self, d_tensor ):
        super().__init__()

        #The square root of the dimension of the key feature vector is stored in the d_tensor attribute
        self.d_tensor = math.sqrt(d_tensor)
        

    # The forward method accepts queries, keys, and values as inputs and returns the output of the attention mechanism
    def forward(self, queries ,keys, values, mask=None):

        #by transposing the keys, we can compute the dot product between the queries and keys via matrix multiplication
        #The queries and keys are multiplied together and divided by the square root of the dimension of the key vector

        keys_transpose=keys.transpose(2,3)
        attention = (queries @ keys_transpose)/self.d_tensor

        #The mask is used to make the attention mechanism causal
        if mask !=None :
            attention = attention.masked_fill(mask == 0, '-inf')

        #The softmax function is applied to the attention weights to obtain the output of the attention mechanism
        #The output of the attention mechanism is multiplied by the values to obtain the final output of the attention mechanism
        attention = torch.softmax(attention, dim=-1)
        values = attention @ values#

        return values, attention#

#Testing the ScaleDotProductAttention class
if __name__ == "__main__":

    #The queries, keys, and values are 4D tensors with the following dimensions: batch size, number of heads, number of timesteps, and dimension features
    batch_size = 2
    heads = 3
    timesteps = 8
    feats = 4

    keys = torch.rand(batch_size, heads, timesteps, feats)
    quereis = torch.rand(batch_size, heads, timesteps, feats)
    values = torch.rand(batch_size, heads, timesteps, feats)

    #d_tensor is the dimension of the key and value feature vectors
    d_tensor = keys.shape[-1]

    test = ScaleDotProductAttention(d_tensor)
    values, attention = test(quereis,keys,values)
    print(values.shape)






    
    
    
    

