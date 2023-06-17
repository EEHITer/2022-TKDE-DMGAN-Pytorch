'''
Date: 2020-10-22 13:56:04
LastEditTime: 2021-01-13 20:24:27
Description: Transformer
FilePath: /DMGAN/model/Transformer.py
'''
import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable

class Mish(nn.Module):

    def __init__(self, inplace=True):
        super().__init__()

        self.inplace = inplace

    def forward(self, x):

        return x * (F.tanh(F.softplus(x)))

class Attention:
    def __init__(self, input_len, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type',)
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', ))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.gc_for_gru = bool(model_kwargs.get("gc_for_gru"))
        self.input_len = input_len

        # Transformer parameters
        self.d_model = self.rnn_units * self.num_rnn_layers  # Embedding Size
        self.d_ff = int(model_kwargs.get('d_ff'))
        self.d_k = int(model_kwargs.get('d_k'))
        self.d_v = int(model_kwargs.get('d_v'))
        self.n_layers = int(model_kwargs.get('n_layers'))
        self.n_heads = int(model_kwargs.get('n_heads'))
        self.tgt_output_dim = int(model_kwargs.get('tgt_output_dim'))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, input_len, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # log -> exp
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2.0) *
                             -(math.log(10000.0) / d_model))
        # position * div_term 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)  # x.size(1)就是有多少个pos
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_maks(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()

    return subsequence_mask

class ScaledDotProductAttention(nn.Module, Attention):
    def __init__(self, input_len,  dropout, adj_mx, **model_kwargs, ):
        super(ScaledDotProductAttention, self).__init__()
        Attention.__init__(self, input_len, adj_mx, **model_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.W = torch.nn.Parameter(torch.randn((self.n_heads, self.input_len, self.d_k * 2), requires_grad=True))
        self.register_parameter("Weight", self.W)
        #self.LayerNorm = LayerNorm(self.d_model)


    def forward(self, Q, K, V, scaled=None, attn_mask=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, len_q, len_k]
        '''
        
        attention = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # attention : [batch_size, n_heads, len_q, len_k]
        if attn_mask:
            attention.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        """weighted attention"""
        atten_cat = torch.matmul(self.W, torch.cat([Q, K], dim=-1).transpose(-1, -2)) / np.sqrt(self.d_k)
        atten_cat = self.softmax(atten_cat)

        attention = attention + atten_cat
        
        context = torch.matmul(attention, V) # [batch_size, n_heads, len_q, d_v]

        return context, attention

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadAttention(nn.Module, Attention):
    def __init__(self, input_len, adj_mx, **model_kwargs):
        #super(MultiHeadAttention, self).__init__()
        nn.Module.__init__(self)
        Attention.__init__(self, input_len, adj_mx, **model_kwargs)
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention(input_len, 0, adj_mx, **model_kwargs)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)
        self.LayerNorm = LayerNorm(self.d_model)
    def forward(self, input_Q, input_K, input_V, attn_mask):

        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V, )
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        
        return self.LayerNorm(output + residual), attn
       # return (output + residual), attn

class PoswiseFeedForwardNet(nn.Module, Attention):
    def __init__(self,  input_len, adj_mx, **model_kwargs):
        #super(PoswiseFeedForwardNet, self).__init__()
        nn.Module.__init__(self)
        Attention.__init__(self, input_len, adj_mx, **model_kwargs)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
        self.LayerNorm = LayerNorm(self.d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.LayerNorm(output + residual) # [batch_size, seq_len, d_model]
        #return output + residual

class EncoderLayer(nn.Module, Attention):
    def __init__(self, input_len, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Attention.__init__(self, input_len, adj_mx, **model_kwargs)
        self.enc_self_attn = MultiHeadAttention(input_len, adj_mx, **model_kwargs)
        self.pos_ffn = PoswiseFeedForwardNet(input_len, adj_mx, **model_kwargs)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module, Attention):
    def __init__(self, input_len, adj_mx, **model_kwargs):
        #super(Encoder, self).__init__()
        nn.Module.__init__(self)
        Attention.__init__(self, input_len, adj_mx, **model_kwargs)
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_vocab_size, d_model),freeze=True)
        self.pos_emb = PositionalEncoding(input_len=input_len, d_model=self.d_model, dropout=0.0, max_len=self.input_len)
        self.layers = nn.ModuleList([EncoderLayer(input_len, adj_mx, **model_kwargs) for _ in range(self.n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = self.pos_emb(enc_inputs)
        enc_self_attn_mask = None

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module, Attention):
    def __init__(self, adj_mx, **model_kwargs):
        super(DecoderLayer, self).__init__()
        Attention.__init__(self, adj_mx, **model_kwargs)
        self.dec_self_attn = MultiHeadAttention(adj_mx, **model_kwargs)
        self.dec_enc_attn = MultiHeadAttention(adj_mx, **model_kwargs)
        self.pos_ffn = PoswiseFeedForwardNet(adj_mx, **model_kwargs)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """

        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, adj_mx, **model_kwargs):
        super(Decoder, self).__init__()
        Attention.__init__(self, adj_mx, **model_kwargs)
        
        self.pos_emb = PositionalEncoding(d_model=self.d_model, dropout=0.0, max_len=self.seq_len)
        self.layers = nn.ModuleList([DecoderLayer(adj_mx, **model_kwargs) for _ in range(self.n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        """
        dec_outputs = self.pos_emb(dec_inputs)
        dec_self_attn_mask = None

        dec_enc_attn_mask = None

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns



class Transformer(nn.Module, Attention):
    def __init__(self, input_len, adj_mx, **model_kwargs):
        #super(Transformer, self).__init__()
        #super().__init__()
        nn.Module.__init__(self)
        Attention.__init__(self, input_len, adj_mx, **model_kwargs)
        self.encoder = Encoder(input_len, adj_mx, **model_kwargs)

    def forward(self, enc_inputs, 
                #dec_inputs
                ):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        dec_inputs: [batch_size, tgt_len]
        '''
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        
        return enc_outputs