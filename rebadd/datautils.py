import os
import random
import pickle
import tqdm
import torch
import numpy as np
import pandas as pd
import selfies as sf


def get_fragment_from_selfies(selfies_iter, use_tqdm=True):
    outputs = []
    
    pbar = tqdm.tqdm(selfies_iter) if use_tqdm else selfies_iter
    
    for selfies in pbar:
        fragments = []
        pos = 0
        symbols = list(sf.split_selfies(selfies))
        L = len(symbols)
        
        b = ''
        while pos < L:
            v = symbols[pos]
            
            ## Case 1: Branch symbol (e.g. v = '[Branch1]')
            if 'ch' == v[-4:-2]:
                ## save
                if len(b) > 0: fragments.append(b)
                ## branch size (Q)
                n = int(v[-2])
                Q = 1 + sf.grammar_rules.get_index_from_selfies(*symbols[pos+1:pos+1+n])
                ## branch
                b = ''.join(symbols[pos:pos+1+n+Q])
                ## save and reset
                fragments.append(b)
                b = ''
            ## Case 2: Ring symbol (e.g. v = '[Ring2]')
            elif 'ng' == v[-4:-2]:
                ## number of symbols for ring size (n)
                n = int(v[-2])
                Q = 0
                ## branch
                b += ''.join(symbols[pos:pos+1+n+Q])
                ## save and reset
                fragments.append(b)
                b = ''
            else:
                b += v
                n = 0
                Q = 0
            ## update pos
            pos += 1 + n + Q
            
        if len(b) > 0:
            fragments.append(b)
            
        outputs.append(fragments)
    return outputs

    
class GeneratorData:
    def __init__(self, pickle_data_path, vocabulary_path, start_token='[sos]', end_token='[eos]', pad_token='[nop]', use_cuda=None, device=None):
        ###########################################
        ## 1. Read a file
        ###########################################
        with open(pickle_data_path, 'rb') as fin:
            data = pickle.load(fin)
        self.data = data
        self.n_data = len(self.data)
        self.max_seqlen = max([len(x) for x in self.data]) + 2 # start_toekn & end_token
        '''
        data = [
                  ['[C][O][C][=C][C]',
                   '[Branch2][Ring1][Ring1][S][=Branch1][C][=O][=Branch1][C][=O][C][C][O][S][=Branch1][C][=O][Branch1][C][O][=O]',
                   '[=C]'],
                  ['[C][C]',
                   '[Branch1][C][C]']
               ]
        '''

        ###########################################
        ## 2. Tokenization
        ###########################################
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.vocabs = [pad_token, start_token, end_token]
        with open(vocabulary_path) as fin:
            vocabs = [line.rstrip() for line in fin.readlines()]
        self.vocabs += vocabs
        
        ###########################################
        ## 3. mapping table
        ###########################################
        self.n_characters = len(self.vocabs)
        self.char2idx = {t:i for i,t in enumerate(self.vocabs)}
        
        ###########################################
        ## 4. CUDA availablity check
        ###########################################
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.device = self.set_device() if device is None else device
        
    def set_device(self):
        return torch.device('cuda') if self.use_cuda else torch.device('cpu')

    def random_chunk(self):
        index = random.randint(0, self.n_data-1)
        return [self.start_token] + self.data[index] + [self.end_token]

    def char_tensor(self, char_list):
        if type(char_list) == str:
            char_list = [char_list]
        
        tensor_ = [self.char2idx.get(c) for c in char_list]
        tensor_ = np.array(tensor_, dtype=np.int64)
        return torch.tensor(tensor_, dtype=torch.long, requires_grad=False, device=self.device)

    def random_training_set(self):
        chunk = self.random_chunk()
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target
        
        
if __name__=='__main__':
    list_of_selfies_strings = [
        '[C][C][N][Branch1][Ring1][C][C][C][=Branch1][C][=O][C][=C][Branch1][C][C][N][=C][S][C][Branch2][Ring1][=Branch2][C][=Branch1][C][=O][N][Ring1][=Branch1][C][Ring1][O][C][=C][C][=C][C][=C][Ring1][=Branch1][O][C][Branch1][C][C][C][=C][C][=C][C]',
    ]
    
    fragments = get_fragment_from_selfies(list_of_selfies_strings)[0]
    
    for i, f in enumerate(fragments):
        print(i, f)