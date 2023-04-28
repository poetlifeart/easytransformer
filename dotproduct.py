import math
import torch
import torch.nn as nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k,q, v, mask=None):

        self.softmax=nn.Softmax(dim=-1)
        d_tensor=k.shape[-1]
        k_t=k.transpose(2,3)

        score=(q @ k_t)/math.sqrt(d_tensor)
        if mask !=None :
            score=score.masked_fill(mask==0, -1e9)

        score=self.softmax(score)
        v=score @ v

        return v, score

k=torch.rand(2, 8, 3, 4)
q=torch.rand(2, 8, 3, 4)
v=torch.rand(2, 8, 3, 4)
test=ScaleDotProductAttention()
t, m=test(k,q,v)
print(t.shape)






    
    
    
    

