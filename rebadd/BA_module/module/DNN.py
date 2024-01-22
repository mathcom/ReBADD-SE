import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_layers(nn.Module):
    def __init__(self):
        super(FC_layers, self).__init__()
        self.fc1 = nn.Linear(512, 256, bias = True)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128, bias = True)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1, bias = True)
        self.dropout_layer = nn.Dropout(p = 0.2)
        
    def forward(self, protein_vector, ligand_vector):   
        x = torch.cat([protein_vector, ligand_vector], dim = 1)  
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.elu(x)
        x = self.dropout_layer(x) 
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.elu(x)
        x = self.dropout_layer(x)
        x = self.fc3(x)  
        return x


class Protein_en(nn.Module):
    def __init__(self, encoder, decoder_bridge, embed): 
        super(Protein_en, self).__init__()
        self.protein_encoder = encoder
        self.protein_decoder_bridge = decoder_bridge
        self.protein_embed = embed
        
    def forward(self, src, src_mask, src_lengths):
        '''
        encoder_hidden: (batch_size, max_seq_len, hidden_size * 2)
        encoder_final: (2, batch_size, hidden_size * 2)
        '''
        encoder_hidden, encoder_final = self.protein_encoder(self.protein_embed(src), src_mask, src_lengths)
        '''
        protein_init: (2, batch_size, hidden_size)
        '''
        protein_init = self.protein_decoder_bridge(encoder_final)
        return protein_init, encoder_hidden


class SMILES_en(nn.Module):
    def __init__(self, encoder, decoder_bridge, embed):  
        super(SMILES_en, self).__init__()
        self.smiles_encoder = encoder
        self.smiles_decoder_bridge = decoder_bridge
        self.smiles_embed = embed
        
    def forward(self, src, src_mask, src_lengths):
        '''
        encoder_hidden: (batch_size, max_seq_len, hidden_size * 2)
        encoder_final: (2, batch_size, hidden_size * 2)
        '''
        encoder_hidden, encoder_final = self.smiles_encoder(self.smiles_embed(src), src_mask, src_lengths)
        '''
        smiles_init: (2, batch_size, hidden_size)
        '''
        smiles_init = self.smiles_decoder_bridge(encoder_final)
        return smiles_init, encoder_hidden
        
        
class regression_model(nn.Module):
    def __init__(self, protein_en, compound_en, fc_layers):
        super(regression_model, self).__init__()
        self.protein_encoder = protein_en
        self.smiles_encoder = compound_en
        self.fc_layers = fc_layers
        
    def forward(self, protein_input, compound_input, protein_reverse, compound_reverse):
        '''
        protein_init: (2, batch_size, hidden_size)
        compound_init: (2, batch_size, hidden_size)
        '''
        protein_init, _ = self.protein_encoder(protein_input.src, protein_input.src_mask, protein_input.src_lengths)
        compound_init, _ = self.smiles_encoder(compound_input.src, compound_input.src_mask, compound_input.src_lengths)
        '''
        protein_vector: (batch_size, 2*hidden_size)
        compound_vector: (batch_size, 2*hidden_size)
        '''
        protein_vector = torch.cat([protein_init[0], protein_init[1]], dim = 1)
        compound_vector = torch.cat([compound_init[0], compound_init[1]], dim = 1)
    
        protein_vector = protein_vector[protein_reverse]
        compound_vector = compound_vector[compound_reverse]
    
        outputs = self.fc_layers(protein_vector, compound_vector)
       
        return outputs 
