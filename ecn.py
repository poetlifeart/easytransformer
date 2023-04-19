import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
 
class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
   
        super(PositionalEncoding, self).__init__()

  # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device, dtype=torch.float64)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device, dtype=torch.float64)
        pos = pos.unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device, dtype=torch.float64)
        _2i=_2i.unsqueeze(dim=0)
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
       
        self.encoding[:, 0::2] = torch.sin(pos @ (10000 ** (_2i / d_model)).pow_(-1))
        
        self.encoding[:, 1::2] = torch.cos(pos @ (10000 ** (_2i / d_model)).pow_(-1))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]  



position= PositionalEncoding (d_model=8, max_len=20, device='cuda')
x=torch.arange(0, 20, dtype=torch.float64)
x=x.unsqueeze(dim=0)
print(position.forward(x).shape)



