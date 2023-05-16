from torch import nn
import torch

from EncoderLayer import EncoderLayer
from transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  hidden_dim=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask=None):
        x = self.emb(x)
       
       

        for layer in self.layers:
            x = layer(x, s_mask)

        return x


if __name__ == '__main__':
    encoder = Encoder(10, 100, 100, 100, 5, 5, 0.1, 'gpu')
    tensor = torch.ones(256, 100, dtype=torch.long)
    tensor = encoder(tensor)
    
      



