import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))




class TransformerModel(nn.Module):
    def __init__(
            self,
            config
    ):
        super(TransformerModel, self).__init__()
        self.sample_hop = config.sample_hop
        self.emb = nn.Linear(config.input_dim, config.hidden_dim)
        self.data_enhance = nn.Dropout(0.8)

        self.layers = nn.ModuleList([
            EncoderLayer(config.hidden_dim, config.hidden_dim * 2, config.dropout_rate, config.attention_dropout_rate, config.num_heads)
            for _ in range(config.n_layers)
        ])
        self.attn_layer = nn.Linear(config.hidden_dim * 2, 1)
        self.for_norm = nn.LayerNorm(config.hidden_dim)
        self.classify = nn.Linear(config.hidden_dim, config.n_class)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()
        self.apply(lambda module: init_params(module, n_layers=config.n_layers))


    def forward(self, batched_data, pe_data, labels=None):

        output = self.emb(batched_data)
        # output = self.data_enhance(output)
        seq_len = (output.shape[1] - 1) // self.sample_hop

        # transformer encoder
        for enc_layer in self.layers:
            output = enc_layer(output, pe_data)

        M_col = torch.cat([torch.arange(self.sample_hop).unsqueeze(0) for _ in range(seq_len)], dim=0).t().reshape(-1)
        M_row = torch.arange(output.shape[1] - 1)
        M = SparseTensor(row=M_row, col=M_col, value=torch.ones((output.shape[1] - 1)), sparse_sizes=(M_row.shape[0], self.sample_hop)).to_dense()

        target = output[:, 0, :].unsqueeze(1).repeat(1, self.sample_hop, 1)
        split_tensor = torch.split(output, [1, output.shape[1]-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
        aggre_tensor = neighbor_tensor.transpose(1, 2) @ M.cuda()
        aggre_tensor = aggre_tensor.transpose(1, 2)



        layer_atten = self.attn_layer(torch.cat((target, aggre_tensor), dim=2))

        layer_atten = F.softmax(layer_atten, dim=1)

        neighbor_tensor = aggre_tensor * layer_atten

        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        neighbor_tensor = self.for_norm(neighbor_tensor)

        output = (node_tensor + neighbor_tensor).squeeze()
        output = F.leaky_relu(output)
        pred = self.classify(output)
        pred = self.dropout(pred)

        if labels is None:
            return F.log_softmax(pred, dim=1)
        else:
            return self.loss(pred, labels), F.log_softmax(pred, dim=1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)

        self.linear_q_pe = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k_pe = nn.Linear(hidden_size, num_heads * att_size)

        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, q_pe, k_pe, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]

        q_pe = self.linear_q_pe(q_pe).view(batch_size, -1, self.num_heads, d_k)
        k_pe = self.linear_k_pe(k_pe).view(batch_size, -1, self.num_heads, d_k)
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2).transpose(2, 3)
        q_pe = q_pe * self.scale
        x_pe = torch.matmul(q_pe, k_pe)

        x = x + x_pe

        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, x_pe, attn_bias=None):

        y = self.self_attention(x, x, x, x_pe, x_pe, attn_bias)
        y = gelu(y)
        y = self.self_attention_dropout(y)
        y = self.self_attention_norm(x + y)

        output = self.ffn(y)
        output = self.ffn_dropout(output)
        output = self.ffn_norm(y + output)

        return output



