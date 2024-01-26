import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rdkit.Chem import MolFromSmiles, MolToSmiles

BA_MODULE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path = sys.path if BA_MODULE_PATH in sys.path else [BA_MODULE_PATH] + sys.path

from module.voca import ProteinVoca, LigandVoca
from module.DNN import Protein_en, SMILES_en, regression_model, FC_layers
from module.RNN import Encoder, BahdanauAttention
from utils import get_regression_result



def build_Protein_en(n_tokens_protein):
    protein_encoder = Encoder(input_size=32, hidden_size=128, num_layers=2, dropout=0.2)
    protein_decoder_bridge = torch.nn.Linear(256, 128)
    protein_embed = torch.nn.Embedding(n_tokens_protein, 32)
    return Protein_en(protein_encoder, protein_decoder_bridge, protein_embed)




def build_SMILES_en(n_tokens_smiles):
    smiles_encoder = Encoder(input_size=32, hidden_size=128, num_layers=2, dropout=0.2)
    smiles_decoder_bridge = torch.nn.Linear(256, 128)
    smiles_embed = torch.nn.Embedding(n_tokens_smiles, 32)
    return SMILES_en(smiles_encoder, smiles_decoder_bridge, smiles_embed)




def build_attention():
    ## context shape: [B, 1, 2D], alphas shape: [B, 1, M]
    return BahdanauAttention(hidden_size=256, key_size=256, query_size=256)




def build_regressor(n_tokens_protein, n_tokens_smiles, use_attention):
    protein_encoder = build_Protein_en(n_tokens_protein)
    smiles_encoder = build_SMILES_en(n_tokens_smiles)
    fc_layers = FC_layers()
    if use_attention:
        attention_qpks = build_attention() # query = protein : key = smiles
        attention_qskp = build_attention() # query = smiles : key = protein
    else:
        attention_qpks = None
        attention_qskp = None
    return regression_model(protein_encoder, smiles_encoder, fc_layers, attention_qpks, attention_qskp)

    
    
    
def normalize_SMILES(smi):
    try:
        mol = MolFromSmiles(smi)
        smi_rdkit = MolToSmiles(
                        mol,
                        isomericSmiles=False,   # modified because this option allows special tokens (e.g. [125I])
                        kekuleSmiles=False,     # default
                        rootedAtAtom=-1,        # default
                        canonical=True,         # default
                        allBondsExplicit=False, # default
                        allHsExplicit=False     # default
                    )
    except:
        smi_rdkit = ''
    return smi_rdkit




class regression_model(torch.nn.Module):
    def __init__(self, protein_en, compound_en, fc_layers, attention_qpks, attention_qskp):
        super(regression_model, self).__init__()
        self.protein_encoder = protein_en
        self.smiles_encoder = compound_en
        self.fc_layers = fc_layers
        self.attention_qpks = attention_qpks
        self.attention_qskp = attention_qskp
        
    def forward(self, protein_input, compound_input, protein_reverse, compound_reverse):
        
        ## protein
        '''
        protein_init: (2, batch_size, hidden_size)
        protein_hidden: (batch_size, max_seq_len, 2*hidden_size)
        protein_vector: (batch_size, 2*hidden_size)
        '''
        protein_init, protein_hidden = self.protein_encoder(protein_input.src, protein_input.src_mask, protein_input.src_lengths)
        protein_vector = torch.cat([protein_init[0], protein_init[1]], dim = 1)
        protein_vector = protein_vector[protein_reverse]

        
        ## smiles
        '''
        compound_init: (2, batch_size, hidden_size)
        compound_hidden: (batch_size, max_seq_len, 2*hidden_size)
        compound_vector: (batch_size, 2*hidden_size)
        '''
        compound_init, compound_hidden = self.smiles_encoder(compound_input.src, compound_input.src_mask, compound_input.src_lengths)
        compound_vector = torch.cat([compound_init[0], compound_init[1]], dim = 1)
        compound_vector = compound_vector[compound_reverse]

        
        ## cross-attention
        '''
        new_protein_vector: (batch_size, 1, 2*hidden_size)
        new_compound_vector: (batch_size, 1, 2*hidden_size)
        '''
        if self.attention_qpks is not None and self.attention_qskp is not None:
            new_protein_vector, _ = self.attention_qpks(protein_vector.unsqueeze(1), compound_hidden)
            new_compound_vector, _ = self.attention_qskp(compound_vector.unsqueeze(1), protein_hidden)
            protein_vector = new_protein_vector.squeeze(1)
            compound_vector = new_compound_vector.squeeze(1)
        
        ## mlp
        outputs = self.fc_layers(protein_vector, compound_vector)
       
        return outputs 




class DTA(object):
    def __init__(self, device, use_cuda, use_attention=False, use_pretrained=True):
        super(DTA, self).__init__()
        ## gpu configs
        self.device = device
        self.use_cuda = use_cuda
        ## ckpt info
        self.filepath_protein_voca = os.path.join(os.path.dirname(__file__), 'ckpt', 'Sequence_voca.txt')
        self.filepath_smiles_voca  = os.path.join(os.path.dirname(__file__), 'ckpt', 'SMILES_voca.txt')
        self.filepath_model = os.path.join(os.path.dirname(__file__), 'ckpt', 'rebadd-dta_merged.pt')
        ## init model
        self.model, self.mc = self._init_model(use_attention)
        self.model.to(self.device)
        ## pretrained
        if use_pretrained:
            _ = self.load(self.filepath_model)
            print(f'[INFO] The pretrained encoders were succeessfully loaded')
        
        
    def __call__(self, aminoseq, list_ligand_seqs, batch_size=500, use_normalization=True):
        ## SMILES normalization
        if use_normalization:
            list_ligand_seqs = [normalize_SMILES(smi) for smi in list_ligand_seqs]
        ## Prepare inputs
        pSeq_SMILES_list = []
        for smi in list_ligand_seqs:
            pSeq_SMILES_list.append((aminoseq, smi))
        ## Pytorch DataLoader
        test_loader = DataLoader(dataset=pSeq_SMILES_list,
                                 batch_size=batch_size,
                                 collate_fn=self.mc,
                                 shuffle=False, pin_memory=False, drop_last=False)
        ## Pytorch Prediction
        self.model.eval()
        list_outs = []
        with torch.no_grad():
            for batch in test_loader:
                out = self.model(*batch) # out.shape = (batch, 1)
                list_outs.append(out)
        scores = torch.cat(list_outs, dim=0)
        scores = scores.view(-1).detach().cpu().numpy() # scores.shape = (batch,)
        return scores
        
    
    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        return self
    

    def _get_dataloader(self, data, batch_size, use_normalization, shuffle=False, drop_last=False):
        list_ligand_seqs = data[0]
        list_target_seqs = data[1]
        list_scores = data[2]
        ## SMILES normalization
        if use_normalization:
            list_ligand_seqs = [normalize_SMILES(smi) for smi in list_ligand_seqs]
        ## Prepare inputs
        pSeq_SMILES_list = []
        for aminoseq, smi, y in zip(list_target_seqs, list_ligand_seqs, list_scores):
            if use_normalization:
                try:
                    smi = normalize_SMILES(smi)
                except:
                    continue
            pSeq_SMILES_list.append((aminoseq, smi, y))
        ## Pytorch DataLoader
        loader = DataLoader(dataset=pSeq_SMILES_list, batch_size=batch_size, collate_fn=self.mc,
                            shuffle=shuffle, pin_memory=False, drop_last=drop_last)
        return loader

    
    def _init_model(self, use_attention):
        ## vocabulary
        protein_voca = ProteinVoca(filepath=self.filepath_protein_voca)
        smiles_voca = LigandVoca(filepath=self.filepath_smiles_voca)
        ## collate function for DataLoader
        mc = Mycall(protein_voca, smiles_voca, self.use_cuda)
        ## Initialize a regressor model
        regressor = build_regressor(protein_voca.num_words, smiles_voca.num_words, use_attention)
        return regressor, mc
    
    
    

class Batch_dnn:
    '''
    For torch.utils.data.DataLoader
    '''
    def __init__(self, src, lengths, USE_CUDA, pad_index = 0):
        ## src: encoder input
        ## trg: decoder input
        ## trg_y: decoder output
        self.src = src
        self.src_lengths = lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()
    
    
    
    
class Mycall:
    '''
    For torch.utils.data.DataLoader
    '''
    def __init__(self, protein_voca, compound_voca, USE_CUDA):
        self.protein_voca = protein_voca
        self.compound_voca = compound_voca
        self.USE_CUDA = USE_CUDA
        
    def __call__(self, batch):
        sampling_protein = [x[0] for x in batch]
        sampling_smiles  = [x[1] for x in batch]

        protein_lines = [self.protein_voca.indexesFromSentence(line) for line in sampling_protein]
        max_protein_len = np.max([len(line) for line in protein_lines])

        smiles_lines = [self.compound_voca.indexesFromSentence(line) for line in sampling_smiles]
        max_smiles_len = np.max([len(line) for line in smiles_lines]) 

        ## prepare the inputs for protein
        protein_input = []
        idx_unk_protein = self.protein_voca.word2index[self.protein_voca.tok_unk]
        idx_pad_protein = self.protein_voca.word2index[self.protein_voca.tok_pad]
        for line in protein_lines:
            A = [self.protein_voca.word2index.get(w, idx_unk_protein) for w in line[:-1]]
            B = [idx_pad_protein for _ in range(max_protein_len - len(line))]
            protein_input.append(A + B)
                
        ## prepare the inputs for compound
        compound_input = []
        idx_unk_compound = self.compound_voca.word2index[self.compound_voca.tok_unk]
        idx_pad_compound = self.compound_voca.word2index[self.compound_voca.tok_pad]
        for line in smiles_lines:
            A = [self.compound_voca.word2index.get(w, idx_unk_compound) for w in line[:-1]]
            B = [idx_pad_compound for _ in range(max_smiles_len - len(line))]
            compound_input.append(A + B)
        
        ## Make input tensors
        protein_input = torch.LongTensor(protein_input)
        compound_input = torch.LongTensor(compound_input)

        ## sort protein sequences
        protein_sorted_lengths = torch.LongTensor([torch.max(torch.nonzero(protein_input[i,:])) + 1 for i in range(protein_input.size(0))])
        protein_sorted_lengths, sorted_idx = protein_sorted_lengths.sort(0, descending=True)
        protein_input = protein_input[sorted_idx]

        ## for reverse sort
        protein_reverse_sort_dict = dict()
        for idx, val in enumerate(sorted_idx):
            protein_reverse_sort_dict[val] = idx 
        protein_reverse_sort_index = np.array([i[1] for i in sorted(protein_reverse_sort_dict.items())])
    
        ## sort SMILES
        compound_sorted_lengths = torch.LongTensor([torch.max(torch.nonzero(compound_input[i,:])) + 1 for i in range(compound_input.size(0))])
        compound_sorted_lengths, sorted_idx = compound_sorted_lengths.sort(0, descending=True)
        compound_input = compound_input[sorted_idx]        

        ## for reverse sort
        compound_reverse_sort_dict = dict()
        for idx, val in enumerate(sorted_idx):
            compound_reverse_sort_dict[val] = idx 
        compound_reverse_sort_index = np.array([i[1] for i in sorted(compound_reverse_sort_dict.items())])
        
        ## Batch_dnn instances
        protein_batch_dnn = Batch_dnn(protein_input, protein_sorted_lengths.tolist(), self.USE_CUDA, idx_pad_protein)
        compound_batch_dnn = Batch_dnn(compound_input, compound_sorted_lengths.tolist(), self.USE_CUDA, idx_pad_compound)
        
        if len(batch[0]) == 3:
            sampling_labels = torch.Tensor(np.array([x[2] for x in batch]))
            if self.USE_CUDA:
                sampling_labels = sampling_labels.cuda()
            return protein_batch_dnn, compound_batch_dnn, protein_reverse_sort_index, compound_reverse_sort_index, sampling_labels
        else:
            return protein_batch_dnn, compound_batch_dnn, protein_reverse_sort_index, compound_reverse_sort_index