import torch
import torch.nn as nn
import scale_dot_product_attention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.linear_keys = nn.Linear(d_model, d_model)
        self.linear_queries = nn.Linear(d_model, d_model)
        self.linear_values = nn.Linear(d_model, d_model)

        self.mix_values = nn.Linear(d_model, d_model)


    def forward(self, keys, queries, values, mask=None):
        print(keys.shape, "keys")
        keys    = self.linear_keys(keys)
        queries = self.linear_queries(queries)
        values  = self.linear_values(values)
        keys, queries, values = self.split(keys), self.split(queries), self.split(values)


        dot=scale_dot_product_attention.ScaleDotProductAttention(self.d_model//self.n_head)
        values, attention = dot(keys, queries, values)
        keys, queries, values = self.concat(keys), self.concat(queries), self.concat(values)

        values = self.mix_values(values)


        return values   


    def split (self, tensor ):
        batch_size, length, d_model = tensor.shape
        d_tensor = d_model // self.n_head
        # we transpose 1 and 2 dimensions to move the the heads so that pytorch batches over heads. 
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat (self, tensor):
        batch_size, n_head, length, d_tensor = tensor.shape
        d_model = n_head*d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


if __name__ == '__main__':
    multihead = MultiHeadAttention(20,5)
    tensor = torch.rand(20,4,20)
    tensor = multihead.split(tensor)
    print(tensor.shape)


    tensor = multihead.concat(tensor)
    print(tensor.shape)


