import math
import torch
from torch import nn
import torch.nn.functional as F
from config import ModelParams


# class TokenEmbeddings(nn.Module):
#     def __init__(self,
#                  pre_trained:torch.Tensor=None,
#                  vocab_size:int=None,
#                  embed_dim:int=None,
#                  pre_trained_freeze:bool=True) -> None:
#         """
#         Parameters:
#             pre_trained: a pre-trained embeddings from Word2Vec model
#             vocab_size: size of vocab. aka: total number of words
#             embed_dim: dimension of embedding. aka: the dimension of the vector that represents each word in the vocab. Ex: 512
#             no_grad: allow or not weights are updated if token_embeddings != None
#         """
#         super().__init__()

#         if pre_trained == None:
#             self.vocab_size = vocab_size
#             self.embed_dim = embed_dim
#             self.token_embeddings = nn.Embedding(
#                 num_embeddings=self.vocab_size,
#                 embedding_dim=self.embed_dim
#             )
#         else:
#             self.vocab_size = pre_trained.size(0)
#             self.embed_dim = pre_trained.size(1)
#             self.token_embeddings = nn.Embedding.from_pretrained(pre_trained, freeze=pre_trained_freeze)

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         return self.token_embeddings(x)
    

class TokenEmbeddings(nn.Module):
    def __init__(self, params:ModelParams) -> None:
        """
        Parameters:
            vocab_size: size of vocab. aka: total number of words
            embed_dim: dimension of embedding. aka: the dimension of the vector that represents each word in the vocab. Ex: 512
        """
        super().__init__()

        self.vocab_size = params.vocab_size
        self.embed_dim = params.embed_dim
        self.token_embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.token_embeddings(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, params:ModelParams) -> None:
        """
        Parameters:
            seq_len: length of input sequence. aka number of words in a sentence
            embed_dim: dimension of embedding. Ex: 512
        """
        super().__init__()

        self.seq_len = params.sequence_len
        self.embed_dim = params.embed_dim

        pos_encoding = torch.zeros(size=(self.seq_len, self.embed_dim), dtype=torch.float32)
        angle_numerator = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(1)
        angle_denominator = torch.pow(10000.0, (2 * (torch.arange(self.embed_dim, dtype=torch.float32) // 2)) / self.embed_dim).unsqueeze(0)
        angles = angle_numerator / angle_denominator

        pos_encoding[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angles[:, 1::2])
        pos_encoding = pos_encoding.unsqueeze(0) # (seq_len, embed_dim) --> (1, seq_len, embed_dim)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        z = self.pos_encoding[:, :x.size(1), :]
        # print("x", x.shape)
        # print("z", z.shape)
        return x + z.requires_grad_(False)
    

class HeadAttention(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 head_dim:int,
                 dropout_prob:float=0.2) -> None:
        """
        Parameters:
            embed_dim: dimension of embedding. Ex: 512
            num_heads: output dim of HeadAttention
        """
        super().__init__()

        self.embed_dim = embed_dim # d_in
        self.head_dim = head_dim # d_out
        self.dropout_prob = dropout_prob

        self.Q = nn.Linear(in_features=self.embed_dim, out_features=self.head_dim, dtype=torch.float32, bias=True)
        self.K = nn.Linear(in_features=self.embed_dim, out_features=self.head_dim, dtype=torch.float32, bias=True)
        self.V = nn.Linear(in_features=self.embed_dim, out_features=self.head_dim, dtype=torch.float32, bias=True)

        self.dropout = nn.Dropout(p=dropout_prob)
    
    def _scaled_dot_product_attention(self, 
                                      query:torch.Tensor, 
                                      key:torch.Tensor, 
                                      value:torch.Tensor, 
                                      mask:torch.Tensor=None):
        dim_k = query.size(-1) # aka. embed_dim
        
        attn_scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(dim_k)
        if mask != None:
            attn_scores.masked_fill_(mask == 0, float("-inf"))
        
        weights = F.softmax(attn_scores, dim=-1)
        if self.dropout > 0:
            weights = self.dropout(weights)
        
        return torch.bmm(weights, value)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        z = self._scaled_dot_product_attention(self.Q(x), self.K(x), self.V(x), mask)
        return z


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 num_heads:int,
                 head_dim:int=None,
                 dropout_prob:float=0.2) -> None:
        """
        Parameters:
            embed_dim: dimension of embedding. Ex: 512
            num_heads: number of HeadAttention
            head_dim: output dim of a HeadAttention. If None, head_dim = embed_dim // num_heads
            dropout_prob: dropout probability
        """
        super().__init__()

        # Make sure embed_dim is divisible by num_heads
        assert (embed_dim % num_heads == 0), "embed_dim is not divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim != None else (embed_dim // num_heads)

        self.heads = nn.ModuleList(
            [
                HeadAttention(embed_dim=self.embed_dim, head_dim=self.head_dim, dropout_prob=dropout_prob)
                for _ in range(self.num_heads)
            ]
        )

        self.out_linear = nn.Linear(in_features=(self.head_dim * self.num_heads), out_features=self.embed_dim, dtype=torch.float32, bias=True)

    # def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        z = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        z = self.out_linear(z)
        return z


class MultiHeadAttentionWeightSplit(nn.Module):
    def __init__(self, params:ModelParams) -> None:
        """
        """
        super().__init__()

        # Make sure embed_dim is divisible by num_heads
        assert (params.embed_dim % params.num_heads == 0), "embed_dim is not divisible by num_heads"

        self.in_dim = params.embed_dim
        self.out_dim = params.embed_dim

        self.num_heads = params.num_heads
        self.head_dim = (params.embed_dim // params.num_heads)
        self.dropout_prob = params.dropout_prob

        self.Q = nn.Linear(in_features=self.in_dim, out_features=self.out_dim, dtype=torch.float32, bias=True)
        self.K = nn.Linear(in_features=self.in_dim, out_features=self.out_dim, dtype=torch.float32, bias=True)
        self.V = nn.Linear(in_features=self.in_dim, out_features=self.out_dim, dtype=torch.float32, bias=True)

        self.out_linear = nn.Linear(in_features=self.out_dim, out_features=self.out_dim, dtype=torch.float32, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def _scaled_dot_product_attention(self, 
                                      query:torch.Tensor, 
                                      key:torch.Tensor, 
                                      value:torch.Tensor, 
                                      mask:torch.Tensor=None):
        
        dim_k = query.size(-1) # aka. head_dim
        # (batch, num_heads, seq_len, head_dim) --> (batch, num_heads, seq_len, seq_len)
        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_k) #torch.bmm() use only for tensor 3D. In this case our tensor is 4D. So, we must use @ instead
        if mask != None:
            attn_scores.masked_fill_(mask == 0, float("-inf"))
        
        weights = F.softmax(attn_scores, dim=-1)        
        if self.dropout_prob > 0:
            weights = self.dropout(weights)

        return (weights @ value) # torch.bmm() use only for tensor 3D

    # def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # (batch, seq_len, in_dim) --> (batch, seq_len, out_dim)
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # Reshape: (batch, seq_len, out_dim) --> (batch, seq_len, num_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, seq_len, num_heads, head_dim) --> (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        z = self._scaled_dot_product_attention(q, k, v, mask)

        # Combine all the heads together
        # (batch, head_dim, seq_len, num_heads) --> (batch, seq_len, head_dim, num_heads) --> (batch, seq_len, out_dim)
        z = z.transpose(1, 2).contiguous().view(batch, seq_len, self.out_dim) # self.out_dim = self.num_heads * self.head_dim
        return self.out_linear(z)


class MultiHeadAttentionKVCache(nn.Module):
    def __init__(self, params:ModelParams) -> None:
        """
        """
        super().__init__()

        # Make sure embed_dim is divisible by num_heads
        assert (params.embed_dim % params.num_heads == 0), "embed_dim is not divisible by num_heads"
        
        self.batch_size = params.train_batch_size
        self.seq_len = params.sequence_len

        self.embed_dim = params.embed_dim
        self.num_heads = params.num_heads # aka. num heads of Query
        self.num_kv_heads = params.num_heads if params.num_kv_heads == None else params.num_kv_heads # num heads of Key & Value
        self.num_repeat = (self.num_heads // self.num_kv_heads)
        self.head_dim = (self.embed_dim // self.num_heads)
        self.dropout_prob = params.dropout_prob

        self.Q = nn.Linear(in_features=self.embed_dim, out_features=(self.num_heads * self.head_dim), dtype=torch.float32, bias=True)
        self.K = nn.Linear(in_features=self.embed_dim, out_features=(self.num_kv_heads * self.head_dim), dtype=torch.float32, bias=True)
        self.V = nn.Linear(in_features=self.embed_dim, out_features=(self.num_kv_heads * self.head_dim), dtype=torch.float32, bias=True)

        self.out_linear = nn.Linear(in_features=(self.num_heads * self.head_dim), out_features=self.embed_dim, dtype=torch.float32, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_prob)

        kv_cache_size = (self.batch_size, self.seq_len, self.num_kv_heads, self.head_dim)
        
        self.register_buffer('k_cache',
                             torch.zeros(kv_cache_size, dtype=torch.float32),
                             persistent=False)
        self.register_buffer('v_cache',
                             torch.zeros(kv_cache_size, dtype=torch.float32),
                             persistent=False)

    def _kv_repeated(x:torch.Tensor, num_repeat:int) -> torch.Tensor:
        if num_repeat == 1:
            return x
        
        batch_size, seq_len, num_kv_heads, head_dim = x.shape
        z = x[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, num_repeat, head_dim)
        return z.reshape(batch_size, seq_len, (num_kv_heads * num_repeat), head_dim)
        
    
    def _scaled_dot_product_attention(self, 
                                      query:torch.Tensor, 
                                      key:torch.Tensor, 
                                      value:torch.Tensor, 
                                      mask:torch.Tensor=None):
        
        dim_k = query.size(-1) # aka. head_dim
        # (batch, num_heads, seq_len, head_dim) --> (batch, num_heads, seq_len, seq_len)
        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_k) #torch.bmm() use only for tensor 3D. In this case our tensor is 4D. So, we must use @ instead
        if mask != None:
            attn_scores.masked_fill_(mask == 0, float("-inf"))
        
        weights = F.softmax(attn_scores, dim=-1)        
        if self.dropout_prob > 0:
            weights = self.dropout(weights)

        return torch.matmul(weights, value) # torch.bmm() use only for tensor 3D

    def forward(self, x:torch.Tensor, input_pos:int, mask:torch.Tensor=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape # (batch, 1, embed_dim)

        # (batch, 1, embed_dim) --> (batch, 1, num_heads * head_dim)
        q = self.Q(x)
        # (batch, 1, embed_dim) --> (batch, 1, num_kv_heads * head_dim)
        k = self.K(x)
        # (batch, 1, embed_dim) --> (batch, 1, num_kv_heads * head_dim)
        v = self.V(x)

        # Reshape:
        # (batch, 1, num_heads * head_dim) --> (batch, 1, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # (batch, 1, num_kv_heads * head_dim) --> (batch, 1, num_kv_heads, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        # (batch, 1, num_kv_heads * head_dim) --> (batch, 1, num_kv_heads, head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Replace the entry in the cache for this token
        self.k_cache[:batch_size, input_pos:(input_pos + seq_len)] = k
        self.v_cache[:batch_size, input_pos:(input_pos + seq_len)] = v

        # Get all the cached of Key and Value
        # (batch_size, seq_len_in_kv, num_kv_heads, head_dim)
        cached_k = self.k_cache[:batch_size, :(input_pos + seq_len)]
        cached_v = self.v_cache[:batch_size, :(input_pos + seq_len)]
        
        # (batch_size, seq_len_in_kv, (num_kv_heads * num_repeat), head_dim)
        cached_k = self._kv_repeated(cached_k, self.num_repeat)
        cached_v = self._kv_repeated(cached_v, self.num_repeat)

        # Transpose: 
        # (batch_size, seq_len, num_heads, head_dim) --> (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        # (batch_size, seq_len_in_kv, (num_kv_heads * num_repeat), head_dim) --> (batch_size, (num_kv_heads * num_repeat), seq_len_in_kv, head_dim)
        cached_k = cached_k.transpose(1, 2)
        # (batch_size, seq_len_in_kv, (num_kv_heads * num_repeat), head_dim) --> (batch_size, (num_kv_heads * num_repeat), seq_len_in_kv, head_dim)
        cached_v = cached_v.transpose(1, 2)

        z = self._scaled_dot_product_attention(q, cached_k, cached_v, mask)

        # Combine all the heads together
        # (batch_size, head_dim, seq_len, num_heads) --> (batch_size, seq_len, head_dim, num_heads) --> (batch_size, seq_len, (num_heads * head_dim))
        z = z.transpose(1, 2).contiguous().view(batch_size, seq_len, (self.num_heads * self.head_dim))
        return self.out_linear(z)


class FeedForward(nn.Module):
    def __init__(self, params:ModelParams) -> None:
        """
        Parameters:
            embed_dim: dimension of embedding. Ex: 512
            ff_dim: Feed forward dimension
            dropout_prob: dropout probability
        """
        super().__init__()

        self.embed_dim = params.embed_dim
        self.ff_dim = params.ff_dim
        self.dropout_prob = params.dropout_prob
        self.activation_name = params.activation_name.lower()

        self.first_linear = nn.Linear(in_features=self.embed_dim, out_features=self.ff_dim, dtype=torch.float32)
        self.last_linear = nn.Linear(in_features=self.ff_dim, out_features=self.embed_dim, dtype=torch.float32)
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        z = self.first_linear(x)

        if self.activation_name == 'relu':
            z = F.relu(z)
        else:
            z = F.gelu(z)

        z = self.last_linear(z)
        if self.dropout_prob > 0:
            z = self.dropout(z)
        return z


class Decoder(nn.Module):
    def __init__(self, params:ModelParams) -> None:
        """
        Parameters:
            embed_dim: dimension of embedding. Ex: 512
            num_heads: number of HeadAttention
            ff_dim: Feed forward dimension
            dropout_prob: dropout probability
            norm_placing: place the normalization layer 'post' or 'pre'
        """
        super().__init__()

        self.embed_dim = params.embed_dim
        self.ff_dim = params.ff_dim
        self.num_heads = params.num_heads
        # self.head_dim = head_dim
        self.dropout_prob = params.dropout_prob
        self.norm_placing = params.norm_placing.lower()

        # self.attn_heads = MultiHeadAttention(embed_dim=self.embed_dim, 
        #                                      num_heads=self.num_heads, 
        #                                      head_dim=self.head_dim, 
        #                                      dropout_prob=self.dropout_prob)
        
        self.attn_heads = MultiHeadAttentionWeightSplit(params)
        
        # self.attn_heads = MultiHeadAttentionKVCache(params)
        
        self.feed_forward = FeedForward(params)
        
        num_norm = 2
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(normalized_shape=self.embed_dim, dtype=torch.float32)
                for _ in range(num_norm)
            ]
        )

    def forward(self, x:torch.Tensor, dec_mask:torch.Tensor) -> torch.Tensor:
        if self.norm_placing == 'post':
            z = x + self.attn_heads(x, dec_mask)
            z = self.layer_norms[0](z)
            z = z + self.feed_forward(z)
            z = self.layer_norms[1](z)
        else: # norm_placing == 'pre'
            z = self.layer_norms[0](x)
            x = x + self.attn_heads(z, dec_mask)               
            z = self.layer_norms[1](x)
            z = x + self.feed_forward(z)
        return z


# class DecoderBlocks(nn.Module):
#     def __init__(self,
#                  num_decoders:int, 
#                  embed_dim:int, 
#                  num_heads:int, 
#                  ff_dim:int, 
#                  head_dim:int=None, 
#                  dropout_prob:float=0.2,
#                  norm_placing:str='post') -> None:
#         """
#         Parameters:
#             num_decoders: number of decoder
#             embed_dim: dimension of embedding. Ex: 512
#             num_heads: number of HeadAttention
#             ff_dim: Feed forward dimension
#             head_dim: output dim of a HeadAttention. If None, head_dim = embed_dim // num_heads
#             dropout_prob: dropout probability
#             norm_placing: place the normalization layer 'post' or 'pre'
#         """
#         super().__init__()

#         self.num_decoders = num_decoders
#         self.embed_dim = embed_dim       
#         self.num_heads= num_heads
#         self.ff_dim = ff_dim
#         self.head_dim = head_dim
#         self.dropout_prob = dropout_prob
#         self.norm_placing = norm_placing

        
#         self.decoders = nn.ModuleList(
#             [
#                 Decoder(
#                     embed_dim=self.embed_dim,
#                     num_heads=self.num_heads,
#                     ff_dim=self.ff_dim,
#                     head_dim=self.head_dim,
#                     dropout_prob=self.dropout_prob,
#                     norm_placing=self.norm_placing)
#                 for _ in range(self.num_decoders)
#             ]
#         )

#     def forward(self, x:torch.Tensor, dec_mask:torch.Tensor=None) -> torch.Tensor:
#         for decoder in self.decoders:
#             x = decoder(x, dec_mask)       
#         return x

    
class TransformerDecoder(nn.Module):
    def __init__(self, params:ModelParams) -> None:
        """
        """
        super().__init__()

        self.seq_len = params.sequence_len
        self.dropout_prob = params.dropout_prob

        self.embeddings = TokenEmbeddings(params)
        
        self.positional_encoding = PositionalEncoding(params)
                        
        self.decoders = nn.ModuleList(
            [
                Decoder(params)
                for _ in range(params.num_decoders)
            ]
        )

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.projection = nn.Linear(in_features=params.embed_dim, out_features=params.vocab_size, dtype=torch.float32)

        
    # def _generate_mask(self, seq_len:int) -> torch.Tensor:
    #     # Mask is the lower triangular part of matrix (includes the diagonal) filled with 1
    #     mask = torch.tril(torch.ones(seq_len, seq_len))
    #     return mask
        
    def forward(self, x:torch.Tensor, dec_mask:torch.Tensor) -> torch.Tensor:
        z = self.embeddings(x)
        z = self.positional_encoding(z)
        
        if self.dropout_prob > 0:
            z = self.dropout(z)

        for decoder in self.decoders:
            z = decoder(z, dec_mask)
       
        # (batch, seq_len, embed_dim) --> (batch, seq_len, vocab_size)
        z = self.projection(z)
        return z