import os
import sys
import numpy as np
from rdkit import Chem

REBADD_LIB_PATH = os.path.abspath(os.pardir)
if REBADD_LIB_PATH not in sys.path:
    sys.path = [REBADD_LIB_PATH] + sys.path

from rebadd.chemutils import SAScorer, JNKScorer, GSKScorer, QEDScorer


def softplus(x, w=None, b=None):
    if w and b:
        return np.log(1. + np.exp(np.multiply(np.add(x, b), w)))
    else:
        return np.log(1. + np.exp(x))
    
    
def hard_sigmoid(x, a=4., b=6.):
    v = np.divide(np.subtract(b, x), np.subtract(b, a))
    return np.minimum(1., np.maximum(0., v))
    
    
class Reward_gsk3(object):
    def __init__(self):
        self.clf_gsk = GSKScorer()
        
    def __call__(self, smiles, debug=False):
        ## Reward
        if Chem.MolFromSmiles(smiles) is not None:
            score_A = self.clf_gsk(smiles)
            reward = self.calc_reward(score_A)
        else:
            reward = score_A = 0.        
        return (reward, score_A) if debug else reward
    
    def calc_reward(self, score_A):
        reward = softplus(score_A, 10, -0.5) # gsk3
        return reward
        
    
class Reward_jnk3(object):
    def __init__(self):
        self.clf_jnk = JNKScorer()
        
    def __call__(self, smiles, debug=False):
        ## Reward
        if Chem.MolFromSmiles(smiles) is not None:
            score_A = self.clf_jnk(smiles)
            reward = self.calc_reward(score_A)
        else:
            reward = score_A = 0.        
        return (reward, score_A) if debug else reward
    
    def calc_reward(self, score_A):
        reward = softplus(score_A, 10, -0.5) # jnk3
        return reward
    
    
class Reward_gsk3_jnk3(object):
    def __init__(self):
        self.clf_gsk = GSKScorer()
        self.clf_jnk = JNKScorer()
        
    def __call__(self, smiles, return_min=False, debug=False):
        ## Reward
        if Chem.MolFromSmiles(smiles) is not None:
            score_A = self.clf_gsk(smiles)
            score_B = self.clf_jnk(smiles)
            reward = min(score_A, score_B) if return_min else self.calc_reward(score_A, score_B)
        else:
            reward = score_A = score_B = 0.        
        return (reward, score_A, score_B) if debug else reward
    
    def calc_reward(self, score_A, score_B):
        reward = softplus(score_A, 10, -0.5) # gsk3
        reward *= softplus(score_B, 10, -0.5) # jnk3
        return reward ** 0.5

    
class Reward_gsk3_jnk3_qed_sa(Reward_gsk3_jnk3):
    def __init__(self):
        super(Reward_gsk3_jnk3_qed_sa, self).__init__()
        self.clf_qed = QEDScorer()
        self.clf_sa = SAScorer()
        
        ## benchmark drug
        self.drug_name = "8515"
        self.drug_smi = "C1=CC=C2C(=C1)C3=NNC4=CC=CC(=C43)C2=O"
        
        ## Debug
        print(f'[DEBUG] GSK3({self.drug_name}) = {self.clf_gsk(self.drug_smi):.3f}')
        print(f'[DEBUG] JNK#({self.drug_name}) = {self.clf_jnk(self.drug_smi):.3f}')
        print(f'[DEBUG] QED({self.drug_name}) = {self.clf_qed(self.drug_smi):.3f}')
        print(f'[DEBUG] SA({self.drug_name}) = {self.clf_sa(self.drug_smi):.3f}')
        
    def __call__(self, smiles, return_min=False, debug=False):
        ## Reward
        if Chem.MolFromSmiles(smiles) is not None:
            score_A = self.clf_gsk(smiles)
            score_B = self.clf_jnk(smiles)
            score_C = self.clf_qed(smiles)
            score_D = self.clf_sa(smiles)
            reward = min(score_A, score_B, score_C, score_D) if return_min else self.calc_reward(score_A, score_B, score_C, score_D)
        else:
            reward = score_A = score_B = score_C = score_D = 0.        
        return (reward, score_A, score_B, score_C, score_D) if debug else reward

    def calc_reward(self, score_A, score_B, score_C, score_D):
        reward = softplus(score_A, 10, -0.5) # gsk3
        reward *= softplus(score_B, 10, -0.5) # jnk3
        reward *= softplus(score_C, 10, -0.5) # qed
        reward *= hard_sigmoid(score_D) # sa
        return reward ** 0.25
    


    
if __name__=="__main__":
    _ = Reward_gsk3_jnk3_qed_sa()