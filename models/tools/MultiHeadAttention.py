# @save
import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        batch_size, channels, height, width = queries.shape
        queries = queries.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        keys = keys.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)
        values = values.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)
        # `queries`，`keys`，`values` 的形状:
        # (`batch_size`，查询或者“键－值”对的个数，`num_hiddens`)
        # `valid_lens`　的形状:
        # (`batch_size`，) 或 (`batch_size`，查询的个数)
        # 经过变换后，输出的 `queries`，`keys`，`values`　的形状:
        # (`batch_size` * `num_heads`，查询或者“键－值”对的个数，
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴 0，将第一项（标量或者矢量）复制 `num_heads` 次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # `output` 的形状: (`batch_size` * `num_heads`，查询的个数，
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # `output_concat` 的形状: (`batch_size`，查询的个数，`num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        output_concat = self.W_o(output_concat)
        output_concat = output_concat.reshape(batch_size, height, width, -1)  # 恢复到 (Batch, Height, Width, Channels)
        output_concat = output_concat.permute(0, 3, 1, 2)
        return output_concat


# @save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状。"""
    # 输入 `X` 的形状: (`batch_size`，查询或者“键－值”对的个数，`num_hiddens`)
    # 输出 `X` 的形状: (`batch_size`，查询或者“键－值”对的个数，`num_heads`，
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出 `X` 的形状: (`batch_size`，`num_heads`，查询或者“键－值”对的个数,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状: (`batch_size` * `num_heads`, 查询或者“键－值”对的个数,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


# @save
def transpose_output(X, num_heads):
    """逆转 `transpose_qkv` 函数的操作。"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of `keys` with `keys.transpose(1,2)`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_utils`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


if __name__ == '__main__':
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
    attention.eval()

    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    attention(X, Y, Y, valid_lens).shape
