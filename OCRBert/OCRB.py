import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
        
# class PositionalEmbedding(nn.Module):

#     def __init__(self, d_model, max_len=512):
#         super().__init__()

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.linear = nn.Linear(2*d_model, d_model)
        
        # Compute the positional encodings once in log space.
        pe_start = torch.zeros(max_len, d_model).float()
        pe_start.require_grad = False
        pe_end = torch.zeros(max_len, d_model).float()
        pe_end.require_grad = False
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('pe_start', pe_start)
        self.register_buffer('pe_end', pe_end)
        self.register_buffer('div_term', div_term)

    def forward(self, positions):
        # positions = torch.from_numpy(positions)
        start = positions[:,0].float().unsqueeze(1)
        end = positions[:,1].float().unsqueeze(1)
        self.pe_start[:positions.size(0),:][:, 0::2] = torch.sin(start * self.div_term)
        self.pe_start[:positions.size(0),:][:, 1::2] = torch.cos(start * self.div_term)
        self.pe_end[:positions.size(0),:][:, 0::2] = torch.sin(end * self.div_term)
        self.pe_end[:positions.size(0),:][:, 1::2] = torch.cos(end * self.div_term)
        pe=torch.cat((self.pe_start[:positions.size(0),:], self.pe_end[:positions.size(0),:]), dim=1)
        pe = pe.unsqueeze(0)
        return self.linear(pe)

class OCRBEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.linear = nn.Linear(2*embed_size, embed_size)
        
    def forward(self, sequence, positions):
        x = self.token(sequence) + self.position(positions)
        return self.dropout(x)
        # x = torch.cat((self.token(sequence), self.position(positions).expand(self.token(sequence).size())), dim=-1)
        # return self.dropout(self.linear(x))

    
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn
    

## Transformer结构
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1,unbiased=True, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class LogNormalization(nn.Module):
    def __init__(self, c=1):
        super(LogNormalization, self).__init__()
        self.c = c

    def forward(self, x):
        return torch.log(x + self.c)

class GaussianNormalization(nn.Module):
    def __init__(self):
        super(GaussianNormalization, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdim=True)
        std = torch.std(x, dim=0, keepdim=True)
        return (x - mean) / (std + 1e-8)

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    
    formula:
            LayerNorm(X+ Multi-Head Attention(X))
            LayerNorm(X+ Feed Forward(X))
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        # self.norm = GaussianNormalization()
        # self.norm = LogNormalization()
        
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        output = sublayer(x)
        if isinstance(output, tuple):
            return self.norm(x + self.dropout(output[0])), output[1]
        else:
            return self.norm(x + self.dropout(output))
        # output = sublayer(self.norm(x))
        # if isinstance(output, tuple):
        #     return x + self.dropout(output[0]), output[1]
        # else:
        #     return x + self.dropout(output)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, x, mask):
        x, weights = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x), weights
    
def _init_ocrb_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight) 

## 整个模型
class OCRB(nn.Module):
    """
    OCRB model : Open Chromatin Regions in BERT model.
    """

    def __init__(self, vocab_size, hidden=48, n_layers=4, attn_heads=4, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = OCRBEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.attn_heads, self.hidden * 4, self.dropout) for _ in range(n_layers)])

        self.apply(_init_ocrb_weights)

    def forward(self, x, positions):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # EM = x>0
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)


        x = self.embedding(x,positions)
        # x = x*EM.unsqueeze(-1).float()
        
        # # weights = []
        # for i, transformer in enumerate(self.transformer_blocks):
        #     x, weights = transformer(x, mask)
        #     x = x*EM.unsqueeze(-1).float()
        #     if i == self.n_layers - 1:
        #         weights=torch.mean(weights, dim=1)
        # return x, weights
        
        weights = []
        for i, transformer in enumerate(self.transformer_blocks):
            x, weight = transformer(x, mask)
            # x = x*EM.unsqueeze(-1).float()
            weights.append(torch.mean(weight, dim=1))
            if i == self.n_layers - 1:
                weights=torch.mean(torch.stack(weights, dim=0),dim=0)
                break
        return x, weights
        
        # for i, transformer in enumerate(self.transformer_blocks):
        #     x, weights = transformer(x, mask)
        #     if i == self.n_layers - 1:
        #         weights=torch.mean(weights, dim=1)
        # return x, weights
    

## 重构
class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
    
class OCRBLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, ocrb: OCRB, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.ocrb = ocrb
        # self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.ocrb.hidden, vocab_size)

    def forward(self, x, positions):
        x, weights = self.ocrb(x,positions)
        return self.mask_lm(x), weights