import math 
import torch
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):#
    def __init__(self, d_tensor ):
        super().__init__()
        self.d_tensor=d_tensor#

    def forward(self, queries ,keys, values, mask=None):#

        
 
        keys_transpose=keys.transpose(2,3)#

        attention=(queries @ keys_transpose)/self.d_tensor#
        if mask !=None :
            attention=attention.masked_fill(mask==0, '-inf')#

        attention=torch.softmax(attention, dim=-1)#
        values=attention @ values#

        return values, attention#

if __name__ == "__main__":


    batch_size = 2
    heads = 3
    timesteps = 8
    feats = 4

  

    keys=torch.rand(batch_size, heads, timesteps, feats)#
    quereis=torch.rand(batch_size, heads, timesteps, feats)#
    values=torch.rand(batch_size, heads, timesteps, feats)#
    d_tensor=keys.shape[-1]#
    test=ScaleDotProductAttention(d_tensor)#
    values, attention=test(quereis,keys,values)#
    print(values.shape)#






    
    
    
    

