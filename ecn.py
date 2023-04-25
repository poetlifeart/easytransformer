import torch
import torch.nn as nn
 
class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        super().__init__()

        self.register_buffer('encoding', torch.zeros((max_len, d_model), dtype=torch.float64))
        pos=torch.arange(0, max_len, dtype=torch.float64)
        _2i=torch.arange(0, d_model, step=2, dtype=torch.float64)
       
        arg=torch.outer(pos, (10000 ** (_2i / d_model)).pow_(-1))
        
        self.encoding[:, 0::2] = torch.sin(arg)
        self.encoding[:, 1::2] = torch.cos(arg)
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        batch_size, seq_len = x.shape
        # [batch_size = 128, seq_len = 30]
        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]  


position= PositionalEncoding (d_model=8, max_len=20)
x=torch.arange(0, 20, dtype=torch.float64)
x=x.unsqueeze(dim=0)
print(position.forward(x).shape)



