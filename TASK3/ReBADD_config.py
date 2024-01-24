import os
import sys
import numpy as np
from rdkit import Chem

REBADD_LIB_PATH = os.path.abspath(os.pardir)
if REBADD_LIB_PATH not in sys.path:
    sys.path = [REBADD_LIB_PATH] + sys.path

from rebadd.bindutils import BAScorerBCL2, BAScorerBCLXL, BAScorerBCLW
    

def softplus(x, w=None, b=None):
    if w and b:
        return np.log(1. + np.exp(np.multiply(np.add(x, b), w)))
    else:
        return np.log(1. + np.exp(x))
    
    
def hard_sigmoid(x, a=4., b=6.):
    v = np.divide(np.subtract(b, x), np.subtract(b, a))
    return np.minimum(1., np.maximum(0., v))
        

class Reward_bcl2_bclxl_bclw(object):
    def __init__(self, use_cuda, device):
        super(Reward_bcl2_bclxl_bclw, self).__init__()
        
        ## Binding Affinity Predictor
        self.calc_ba_bcl2 = BAScorerBCL2(use_cuda=use_cuda, device=device)
        self.calc_ba_bclxl = BAScorerBCLXL(use_cuda=use_cuda, device=device)
        self.calc_ba_bclw = BAScorerBCLW(use_cuda=use_cuda, device=device)
        
        ## benchmark drug
        self.drug_name = "navitoclax"
        self.drug_smi = "CC1(CCC(=C(C1)CN2CCN(CC2)C3=CC=C(C=C3)C(=O)NS(=O)(=O)C4=CC(=C(C=C4)NC(CCN5CCOCC5)CSC6=CC=CC=C6)S(=O)(=O)C(F)(F)F)C7=CC=C(C=C7)Cl)C"
        
        ## Debug
        print(f'[DEBUG] BA({self.drug_name},{self.calc_ba_bcl2.pid}) = {self.calc_ba_bcl2(self.drug_smi):.3f}')
        print(f'[DEBUG] BA({self.drug_name},{self.calc_ba_bclxl.pid}) = {self.calc_ba_bclxl(self.drug_smi):.3f}')
        print(f'[DEBUG] BA({self.drug_name},{self.calc_ba_bclw.pid}) = {self.calc_ba_bclw(self.drug_smi):.3f}')

        
    def __call__(self, smiles, return_min=False, debug=False):
        ## Binding affinity
        if Chem.MolFromSmiles(smiles) is not None:
            ba_A = self.calc_ba_bcl2(smiles)
            ba_B = self.calc_ba_bclxl(smiles)
            ba_C = self.calc_ba_bclw(smiles)
        else:
            ba_A = ba_B = ba_C = 0.
        ## normalization
        score_A, score_B, score_C = self.calc_score(ba_A, ba_B, ba_C)
        ## Reward
        if return_min:
            reward = min(score_A, score_B, score_C)
        else:
            reward = self.calc_reward(score_A, score_B, score_C)
        return (reward, ba_A, ba_B, ba_C, score_A, score_B, score_C) if debug else reward
        
    
    def calc_score(self, ba_A, ba_B, ba_C):
        score_A = softplus(ba_A, 2.0, -7.5) # bcl2
        score_B = softplus(ba_B, 2.0, -7.5) # bclxl
        score_C = softplus(ba_C, 2.0, -7.0) # bclw
        return score_A, score_B, score_C
        
        
    def calc_reward(self, score_A, score_B, score_C):
        score = score_A * score_B * score_C
        return score ** 0.333333
    
    
if __name__=="__main__":
    import torch
    use_cuda = False
    device = torch.device('cpu')
    _ = Reward_bcl2_bclxl_bclw(use_cuda=use_cuda, device=device)