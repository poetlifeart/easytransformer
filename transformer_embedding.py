"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import torch

from positional_encoding import PositionalEncoding
from token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        print(x.shape, "x")
        tok_emb = self.tok_emb(x)
        print(tok_emb.shape, "transformer embedding")
        pos_emb = self.pos_emb(x)
        
        result = torch.add(pos_emb, tok_emb)
        
        return self.drop_out(tok_emb + pos_emb)
