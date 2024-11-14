# add all your Encoder and Decoder code here

import torch
from torch import nn
import math

# ---------- Positional Encoding Functions ------------
def sin_cos_func(token_index, list_position, total_dimensions=4, n=100):
    i = list_position // 2
    den = n**((2*i)/ total_dimensions)
    if (list_position % 2 == 0):
        # Do sin if even
        return math.sin(token_index / den)
    else:
        # Do cos if odd
        return math.cos(token_index / den)
    
def positionally_encode_token(token_position, total_dimensions, n):
    token_list = []
    for current_dimension in range(total_dimensions):
        token_list.append(sin_cos_func(token_position, current_dimension, total_dimensions, n))
    return torch.tensor(token_list, dtype=torch.float)

def positionally_encode_batch(batch, n_embd):
    n = 10000
    for sentence_spot in range(len(batch)):
        for token_spot in range(len(batch[sentence_spot])):
            batch[sentence_spot][token_spot] = batch[sentence_spot][token_spot] + positionally_encode_token(token_spot, n_embd, n)
    return batch

# Generates the alibi matrix that is T x T
def get_alibi_matrix(T):
    m = 1
    alibi_list = []
    for i in range(0, -T, -1):
        alibi_list.append(torch.arange(i, i + T))
    alibi_tensor = torch.stack(alibi_list)
    alibi_tensor = alibi_tensor.masked_fill(torch.tril(torch.ones(T, T)) == 0, 0)
    return(alibi_tensor * m)



# ---------- Encoder (some classes reused in decoder) ------------
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, embedding_strategy='basic_positional', mask=False):
        super().__init__()
        self.n_embd = n_embd
        self.key = nn.Linear(n_embd, head_size, bias=False) # Collapses C down to size head_size
        self.query = nn.Linear(n_embd, head_size, bias=False) # Collapses C down to size head_size
        self.value = nn.Linear(n_embd, head_size, bias=False) # Collapses C down to size head_size
        self.embedding_strategy = embedding_strategy
        self.mask = mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B x T x head_size
        q = self.query(x) # B x T x head_size
        v = self.value(x) # B x T x head_size

        if (self.embedding_strategy == 'alibi'):
            attention = (q @ k.transpose(-2, -1) + get_alibi_matrix(T)) * (C**-0.5) # Need to transpose k in order to multiply: result is B x T x T
        else:
            attention = q @ k.transpose(-2, -1) * (C**-0.5) # Need to transpose k in order to multiply: result is B x T x T

        # This part masks all of the tokens before it
        if (self.mask == True):
            attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Fill the 0 values with -inf

        attention = nn.functional.softmax(attention, dim=-1) # Normalize
        output = attention @ v # B x T x head_size
        return output, attention


# Run multiple heads, then concatenate the results and project to the C dimension
class MultipleHeads(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, embedding_strategy, mask):
        super().__init__()
        self.head_list = nn.ModuleList(Head(head_size, n_embd, block_size, embedding_strategy, mask) for i in range(num_heads))
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        head_results = []
        attention_maps = []
        for head in self.head_list:
            output, attention = head.forward(x)
            head_results.append(output)
            attention_maps.append(attention)

        # Concatenates on the C dimension
        all_heads = torch.cat(head_results, dim=-1)
        result = self.dropout(self.proj(all_heads))

        return result, attention_maps


class Encoder_Block(nn.Module):
    def __init__(self, head_size, num_heads, embedding_dimension, block_size, embedding_strategy, n_hidden):
        super().__init__()
        # Each of these layers maintains the shape of the tensor
        self.heads_layer = MultipleHeads(num_heads, head_size, embedding_dimension, block_size, embedding_strategy, mask=False)
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimension, n_hidden), 
            nn.ReLU(),
            nn.Linear(n_hidden, embedding_dimension), 
        )

    def forward(self, x):
        # Shape of x: B x T x C

        #---Attention---
        pre_attention = self.layer_norm_1(x)
        post_attention, attention_maps = self.heads_layer.forward(pre_attention) # Output Shape: B x T x head_size
        x = x + post_attention

        #---Feed Forward---
        pre_ff = self.layer_norm_2(x)
        post_ff = self.feed_forward(pre_ff)
        x = x + post_ff        
        
        return x, attention_maps # Shape of x should be B x T x C


# B = batch size
# T = # tokens
# C = embedding dimensions
class Transformer_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, head_size, num_heads, num_transformer_layers, block_size, embedding_strategy, n_hidden):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.block_size = block_size
        self.basic_positional_embedding = nn.Embedding(block_size, embedding_dimension)
        self.encoder_block_layer = nn.ModuleList(Encoder_Block(head_size, num_heads, embedding_dimension, block_size, embedding_strategy, n_hidden) for _ in range(num_transformer_layers))

    def forward(self, x):
        # Initial input is batches of indices: B x T
        # For each batch, get the embedding for each sentence
        x = self.embedding(x) # After embedding B x T x C
        positional_embeddings = self.basic_positional_embedding(torch.arange(self.block_size))
        x = x + positional_embeddings
        for block in self.encoder_block_layer:
            x, attention_map = block.forward(x)

        return x, attention_map # we only care about the last attention maps


class Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, head_size, num_heads, transformer_layers, n_output, n_hidden, block_size, embedding_strategy='basic_positional'):
        super().__init__()
        self.tranformer_encoder = Transformer_Encoder(vocab_size, embedding_dimension, head_size, num_heads, transformer_layers, block_size, embedding_strategy, n_hidden)
        self.classifier_layer = nn.Sequential(
            nn.Linear(embedding_dimension, n_hidden), 
            nn.Linear(n_hidden, n_output), # 3 Choices in Classification Task
            # Does not use softmax becuase we are using cross entropy loss
        )

    def forward(self, x):
        # Initial input is batches of indices: B x T
        x, _ = self.tranformer_encoder.forward(x) # Output is  B x T x C, don't need attention maps here
        x = torch.mean(x, dim=1) # Averages tokens together for each sentence: B x C
        x = self.classifier_layer(x) # Output shape: B x n_output(3)
        return x

    def get_transformer(self):
        return self.tranformer_encoder


# --------------- Decoder -------------------
class Decoder_Block(nn.Module):
    def __init__(self, head_size, num_heads, embedding_dimension, block_size, hidden_size, embedding_strategy):
        super().__init__()
        # Each of these layers maintains the shape of the tensor
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.masked_heads_layer = MultipleHeads(num_heads, head_size, embedding_dimension, block_size, embedding_strategy, mask=True)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dimension), 
            nn.Dropout(.1)
        )

    def forward(self, x):
        # Shape of x: B x T x C

        #---Attention---
        pre_attention = self.layer_norm_1(x)
        post_attention, attention_maps = self.masked_heads_layer.forward(pre_attention) # Output Shape: B x T x head_size
        x = x + post_attention

        #---Feed Forward---
        pre_ff = self.layer_norm_2(x)
        post_ff = self.feed_forward(pre_ff)
        x = x + post_ff

        return x, attention_maps # Shape of x should be B x T x C


# B = batch size
# T = # tokens
# C = embedding dimensions
# Input is batch of indices B x T
class Transformer_Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, head_size, num_heads, num_transformer_layers, block_size, hidden_layer, embedding_strategy='basic_positional'):
        super().__init__()
        self.embedding_strategy = embedding_strategy
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.block_size = block_size
        self.basic_positional_embedding = nn.Embedding(block_size, embedding_dimension)
        self.decoder_block_layer = nn.ModuleList(Decoder_Block(head_size, num_heads, embedding_dimension, block_size, hidden_layer, embedding_strategy) for _ in range(num_transformer_layers)) # B x T x C
        self.final_linear_layer = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, vocab_size)
            # Not using softmax because we're doing cross entropy loss
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        # Initial input is batches of indices: B x T
        # For each batch, get the embedding for each sentence
        x = self.embedding(x) # After embedding B x T x C

        if (self.embedding_strategy == 'basic_positional'):
            # Basic Positional Encoding using nn.Embedding
            positional_embeddings = self.basic_positional_embedding(torch.arange(self.block_size))
            x = x + positional_embeddings
        elif (self.embedding_strategy == 'sinusoidal'):
            # Sin/Cos Positional Encoding
            x = positionally_encode_batch(x, self.embedding_dimension) # After positional embedding B x T x C
        # else if alibi, do not embed positions

        for block in self.decoder_block_layer:
            x, attention_map = block.forward(x)

        x = self.final_linear_layer(x)
 
        # Two versions of the function
        # Return x and the attention map for training and sanity check
        if (targets == None):
            return x, attention_map
        
        # Just return the loss for the perplexity function
        else:
            # Compute loss if given
            B, T, C = x.shape
            loss = self.loss_function(x.view(B*T, C), targets.view(B*T))
            return loss

