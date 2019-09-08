import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os, math, copy, time, operator
from crf import CRF
import constant

def make_model(src_vocab, trg_entity, trg_intent, pad_idx, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, cuda=True):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = TransformerCrf(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        Generator(d_model, trg_intent, multi=False),
        Generator(d_model, trg_entity),
        CRF(trg_entity, pad_idx, batch_first=True, reduction= 'none', cuda=cuda))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class TransformerCrf(nn.Module):
    """
    A standard Encoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, src_embed, generator, generator2, crf):
        super(TransformerCrf, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.int_cls = generator
        self.ent_cls = generator2
        self.crf = crf

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def forward(self, src, trg, src_mask=None, turn=0):
        "Take in and process masked src and target sequences."
        output = self.encode(src, src_mask)
        pred_int = F.log_softmax(self.int_cls(output), dim=-1)
        output_ent = self.ent_cls(output)
        if src_mask is not None:
            src_mask = src_mask.squeeze(1)[:,1:]
        llh_ent = -self.crf(output_ent, trg, src_mask) 
        pred_ent = self.crf.decode(output_ent, src_mask)
        return pred_ent, llh_ent, pred_int
    
    def decode(self, src, src_mask=None, turn=0):
        output = self.encode(src, src_mask)
        pred_int = F.log_softmax(self.int_cls(output), dim=-1)
        output_ent = self.ent_cls(output)
        if src_mask is not None:
            src_mask = src_mask.squeeze(1)[:,1:]
        pred_ent = self.crf.decode(output_ent, src_mask)
        return pred_ent, pred_int

class TransEncoder(nn.Module):
    """
    A standard Encoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, src_embed, generator, generator2):
        super(TransEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.ent_cls = generator
        self.int_cls = generator2

    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        output = self.encode(src, src_mask)
        pred_ent = F.log_softmax(self.ent_cls(output), dim=-1)
        pred_int = F.log_softmax(self.int_cls(output), dim=-1)
        return pred_ent, pred_int

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, multi=True):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.multi = multi

    def forward(self, x):
        if self.multi:
            return self.proj(x[:,1:])
        else:
            return self.proj(x[:,0])


def clones(module, N):
    "Produce N identical layers."
    #Python list와 ModuleList 차이
    #Parameter hidden or not
    #https://discuss.pytorch.org/t/the-difference-in-usage-between-nn-modulelist-and-python-list/7744
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # -1 represent the last dimension
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        #clone N different weight layers
        #and one unique norm layer
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."

        #residual connection resolve vanishing gradient problem
        #https://arxiv.org/pdf/1603.05027.pdf
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        #Why deepcopy the SublayerConnection layers?
        #To make different batch nomalization parameters for each sublayer
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            # query shape: (batch, h heads, time_step, dimension)
            # mask.unsqueeze(1) for broad casting to h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        #position * div_term shape: (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe.unsqueeze(0) for broadcast
        #x shape = (batch, time_step, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
