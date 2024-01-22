import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def make_model(vocab_size, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1, USE_CUDA = True):
    
    attention = BahdanauAttention(hidden_size)
    
    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(vocab_size, emb_size),
        Generator(hidden_size, vocab_size))
    
    return model.cuda() if USE_CUDA else model


def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=2, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)
    
    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden)

            ## we predict from the pre-output layer, which is
            ## a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    ## cut off everything starting from </s> 
    ## (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)


def lookup_words(x, vocab=None):

    if vocab is not None:
        x = [vocab.index2word[i] for i in x]

    return [str(t) for t in x]


class Batch:
    
    def __init__(self, src, trg, lengths, USE_CUDA, pad_index = 0):
        
        ## src: encoder input
        ## trg: decoder input
        ## trg_y: decoder output
        self.src = src
        self.src_lengths = lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = src
        self.trg_lengths = lengths
        self.trg_y = trg
        self.trg_mask = (self.trg_y != pad_index)
        self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()
            
            self.trg = self.trg.cuda()
            self.trg_y = self.trg_y.cuda()
            self.trg_mask = self.trg_mask.cuda()


class SimpleLossCompute:
    
    def __init__(self, generator, criterion, opt = None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),y.contiguous().view(-1))
        
        loss = loss / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.data.item() * norm


class Generator(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias = False)
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim = -1)        
        
        
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed = embed
        self.generator = generator
    
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
        
    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.embed(src), src_mask, src_lengths)
        
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
        decoder_hidden = None):
        return self.decoder(self.embed(trg), encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden = decoder_hidden)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first = True,
            bidirectional = True, dropout = dropout)    
            
    def forward(self, x, mask, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first = True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first = True)

        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]

        final = torch.cat([fwd_final, bwd_final], dim=2)

        return output, final    


class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                bridge=True):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.rnn = nn.GRU(emb_size + 2*hidden_size, hidden_size, num_layers,
                batch_first = True, dropout = dropout)
                
        self.bridge = nn.Linear(2*hidden_size, hidden_size, bias = True) if bridge else None
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size, 
                                    hidden_size, bias = False)
                                    
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        query = hidden[-1].unsqueeze(1)
        context, attn_probs = self.attention(
            query = query, proj_key = proj_key,
                value = encoder_hidden, mask = src_mask)

        rnn_input = torch.cat([prev_embed, context], dim = 2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim = 2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)
        
        return output, hidden, pre_output
        
    def forward(self, trg_embed, encoder_hidden, encoder_final,
        src_mask, trg_mask, hidden = None, max_len = None):
        
        if max_len is None:
            max_len = trg_mask.size(-1)
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
            
        proj_key = self.attention.key_layer(encoder_hidden)
        
        decoder_states = []
        pre_output_vectors = []
        
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)
            
        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        if encoder_final is None:
            return None  # start with zeros
        return torch.tanh(self.bridge(encoder_final))  


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        ## We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size
        #self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        ## to store attention scores
        self.alphas = None
        
    def forward(self, query, proj_key, value=None, mask=None):
        #assert mask is not None, "mask is required"
        if value is None:
            value = proj_key

        ## We first project the query (the decoder state).
        ## The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        ## Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        ## Mask out invalid positions.
        ## The mask marks valid positions so we invert it using `mask & 0`.
        if mask is not None:
            scores.data.masked_fill_(mask == 0, -float('inf'))
        
        ## Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        ## The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        ## context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas
