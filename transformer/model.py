# -*- coding: utf-8 -*-
import torch
from torch import nn
import math

# --------Encoder----------
class InputEmbedding(nn.Module):
    # d_model:token embedding
    # vocab_size:size of vocab
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch,seq_len)---->(batch,seq_len,d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # d_model:token embedding
    # seq_len:句子的最大长度，为每个位置分配一个向量
    # dropout:添加dropout防止过拟合
    # 位置信息只计算一次，在之后训练中用在每个句子上
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # 创建维度为(seq_len,d_model)的矩阵
        pe = torch.zeros(seq_len, d_model)
        # 创建维度为(seq_len)的向量，元素取值为[0,seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len,1)
        # 创建维度为(d_model)的向量
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 偶数项应用sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数项应用cos
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为每个位置编码添加batch维度，以方便一批一批的使用句子
        pe = pe.unsqueeze(0)  # 在第0维新加了一个维度

        # 当有一个表示模型状态的tensor，要存放模型里，需要注册一个buffer，buffer会跟模型文件一起保存在模型里
        self.register_buffer("pe", pe)

    def forward(self, x):
        # requires_grad_(False)告诉模型训练过程中不需要学习这一部分位置编码，因为是固定的
        # x_shape:[batch seq_len d_model]
        # 取shape[1]是对序列长度进行位置编码
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    # eps:归一化的参数epsilon
    def __init__(self, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # 乘数
        self.bias = nn.Parameter(torch.zeros(1))  # 加数

    # x_shape:[batch seq_len hidden_size]
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # (batch,seq_len,1)
        std = x.std(dim=-1, keepdim=True)  # (batch,seq_len,1)
        # eps保持数值稳定性，防止除数过小
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    # d_ff是内部隐藏层维度
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # [batch,seq_len,d_model]->[linear1/relu/dropout]->[batch,seq_len,d_ff]->[linear2]->[batch,seq_len,d_model]
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        # d_k等于d_model除以h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # view用于重塑形状，transpose用于交换维度
        # (batch, seq_len, d_model) -[view]-> (batch, seq_len, h, d_k) -[transpose]-> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


# Add&Norm
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # 返回一个列表，包括两层残差
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    # src_mask应用在encoder输入，用于屏蔽[pad]填充词对正常词影响
    # 算法流程：x先进行layernorm，然后多头注意力，结果再加上残差；
    # 再对输入进行layernorm，然后送入ffn，输出再加上残差。
    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# -------Decoder---------
class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    # tgt_mask:target mask
    # src_mask:source_mask
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# ------Linear&Softmax--------
# ProjectionalLayer是投影映射层
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # [batch,seq_len,d_model]->[batch,seq_len,vocab_size]
        # dim=-1是指在最后一个维度上
        return torch.log_softmax(self.proj(x), dim=-1)


# ---------transformer----------
class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


# --------初始化----------
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff=2048,
) -> Transformer:
    """
    构造Transformer，创建超参数，初始化参数。

    参数: 
        src_vocab_size: 源语言的词表大小；
        tgt_vocab_size: 目标语言的词表大小；
        src_seq_len: 源语言语言句子最大长度；
        tgt_seq_len: 目标语言句子最大长度；
        （本文的范例为英语到意大利语的翻译，源和目标语言句子长度设置为相同大小。
        但对于其他语言，可能存在源和目标语言相差非常大的情况，届时将这两个值设置为不同大小）
        d_model: 模型大小，与论文一致预设为512；
        N: encoder/decoder的数量，与论文一致预设为6；
        h: 注意力头数，与论文一致预设为8；
        dropout: 与论文一致预设为0.1；
        d_ff: 前馈网络大小，与论文一致预设为2048。
    返回:
        一个完成了参数初始化的Transformer。
    """
    # 初始化embedding层
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # 初始化Positional Embedding层，实际上位置信息的编码只需要一组参数计算一次即可，这里使用相同的参数计算两层是为了更好的解释模型
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # 初始化N个Encoder
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # 初始化N个Decoder
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # 构造Encoder和Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # 初始化投影层
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # 构造transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # 初始化超参
    # xavier_uniform_是初始化神经网络参数的一种方法
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
