import os
import sys
import requests
import torch
import pickle
from rdkit.Chem import MolFromSmiles, MolToSmiles
from torch.utils.data import DataLoader

BA_MODULE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path = sys.path if BA_MODULE_PATH in sys.path else [BA_MODULE_PATH] + sys.path

from BA_module import DTA, normalize_SMILES



class BAScorer:
    def __init__(self, device=None, use_cuda=None):
        self.use_cuda = use_cuda
        self.device = device
        self.model = DTA(device=device, use_cuda=use_cuda)
        
    def __call__(self, smi):
        smi = normalize_SMILES(smi)
        
        score = 0.
        if smi != '':
            with torch.no_grad():
                out = self.model(self.pseq, [smi,], batch_size=1)  # out.shape = (batch, 1)
            score = out.item()
        return score


class BAScorerBCL2(BAScorer):
    def __init__(self, *args, **kwargs):
        super(BAScorerBCL2, self).__init__(*args, **kwargs)
        self.pseq = "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMRPLFDFSWLSLKTLLSLALVGACITLGAYLGHK"
        self.pid = "P10415"
    

class BAScorerBCLXL(BAScorer):
    def __init__(self, *args, **kwargs):
        super(BAScorerBCLXL, self).__init__(*args, **kwargs)
        self.pseq = "MSQSNRELVVDFLSYKLSQKGYSWSQFSDVEENRTEAPEGTESEMETPSAINGNPSWHLADSPAVNGATGHSSSLDAREVIPMAAVKQALREAGDEFELRYRRAFSDLTSQLHITPGTAYQSFEQVVNELFRDGVNWGRIVAFFSFGGALCVESVDKEMQVLVSRIAAWMATYLNDHLEPWIQENGGWDTFVELYGNNAAAESRKGQERFNRWFLTGMTVAGVVLLGSLFSRK"
        self.pid = "Q07817"
        
        
class BAScorerBCLW(BAScorer):
    def __init__(self, *args, **kwargs):
        super(BAScorerBCLW, self).__init__(*args, **kwargs)
        self.pseq = "MATPASAPDTRALVADFVGYKLRQKGYVCGAGPGEGPAADPLHQAMRAAGDEFETRFRRTFSDLAAQLHVTPGSAQQRFTQVSDELFQGGPNWGRLVAFFVFGAALCAESVNKEMEPLVGQVQEWMVAYLETQLADWIHSSGGWAEFTALYGDGALEEARRLREGNWASVRTVLTGAVALGALVTVGAFFASK"
        self.pid = "Q92843"


## DEBUG
if __name__=='__main__':
    ## torch setup
    device = torch.device('cpu')
    use_cuda = False
    
    ## drug
    drugs = {
        'navitoclax':'CC1(CCC(=C(C1)CN2CCN(CC2)C3=CC=C(C=C3)C(=O)NS(=O)(=O)C4=CC(=C(C=C4)NC(CCN5CCOCC5)CSC6=CC=CC=C6)S(=O)(=O)C(F)(F)F)C7=CC=C(C=C7)Cl)C',
        'abt737':'CN(C)CCC(CSC1=CC=CC=C1)NC2=C(C=C(C=C2)S(=O)(=O)NC(=O)C3=CC=C(C=C3)N4CCN(CC4)CC5=CC=CC=C5C6=CC=C(C=C6)Cl)[N+](=O)[O-]'
    }
    
    ## DTA
    calc_affinity_score_bcl2 = BAScorerBCL2(device=device, use_cuda=use_cuda)
    calc_affinity_score_bclxl = BAScorerBCLXL(device=device, use_cuda=use_cuda)
    calc_affinity_score_bclw = BAScorerBCLW(device=device, use_cuda=use_cuda)
    
    for drugname, smi in drugs.items():
        ## evaluation
        score_bcl2 = calc_affinity_score_bcl2(smi)
        score_bclxl = calc_affinity_score_bclxl(smi)
        score_bclw = calc_affinity_score_bclw(smi)
        
        ## results
        print(f'BA({drugname},Bcl-2):\t{score_bcl2:.3f}')
        print(f'BA({drugname},Bcl-xl):\t{score_bclxl:.3f}')
        print(f'BA({drugname},Bcl-w):\t{score_bclw:.3f}')
    