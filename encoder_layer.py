from torch import nn
import torch
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, n_head, drop_prob=0.1):    
        super().__init__()
        self.attention=MultiHeadAttention(d_model, n_head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn=PositionwiseFeedForward(d_model, hidden_dim)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)


    def forward(self, x, mask=None):
        resx = x
        x = self.attention(keys=x, queries=x, values=x)
        x = self.dropout2(x)
        x = x + resx
        x = self.layer_norm1(x)

        resx = x
        self.ffn(x)
        x=self.dropout2(x)
        x = x + resx
        x = self.layernorm2(x)

        return x



if __name__ == '__main__':
    encoder = EncoderLayer(10, 10, 5)
    tensor = torch.ones(5, 10, 10)
    tensor = encoder(tensor)
    print()
    print(tensor.shape)

