import os
import sys
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import TPSA
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import QED
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity

FEASIBILITY_MODULE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path = sys.path if FEASIBILITY_MODULE_PATH in sys.path else [FEASIBILITY_MODULE_PATH] + sys.path
from SA_module.sascorer import readFP2Score, calculateScore
from RA_module.RAscore_XGB import RAScorerXGB
from GSK3_module.utils import gsk3_model
from JNK3_module.utils import jnk3_model


class JNKScorer:
    def __init__(self):
        '''
        RandomForestClassifier(bootstrap=True, ccp_alpha=None, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
        '''
        self.clf = jnk3_model()
        
    def __call__(self, smi):
        mol = MolFromSmiles(smi)
        return self.clf(mol) if mol else 0.
    
    def set_params(self, **kwargs):
        _ = self.clf.clf.set_params(**kwargs)
        return self


class GSKScorer:
    def __init__(self):
        '''
        RandomForestClassifier(bootstrap=True, ccp_alpha=None, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
        '''
        self.clf = gsk3_model()
        
    def __call__(self, smi):
        mol = MolFromSmiles(smi)
        return self.clf(mol) if mol else 0.
    
    def set_params(self, **kwargs):
        _ = self.clf.clf.set_params(**kwargs)
        return self


class SAScorer:
    def __init__(self):
        self._fpscores = readFP2Score()
        
    def __call__(self, smi):        
        try:
            mol = MolFromSmiles(smi)
            return calculateScore(mol, self._fpscores) # range : 1 (easy to make) ~ 10 (very difficult to make)
        except:
            return 10.


class RAScorer:
    def __init__(self):
        '''
        XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.19984033197055842, max_delta_step=0, max_depth=19,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=97, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
        '''
        self.clf = RAScorerXGB()
        
    def __call__(self, smi):
        try:
            return self.clf.predict(smi)
        except:
            return 0.
        
    def set_params(self, **kwargs):
        _ = self.clf.xgb_model.set_params(**kwargs)
        return self
            

class QEDScorer:
    def __init__(self):
        pass
    
    def __call__(self, smi):
        try:
            mol = MolFromSmiles(smi)
            return QED.qed(mol) # quantitative estimation of drug-likeness: ranges between 0 and 1, with 1 being the most drug-like.
        except:
            return 0.
        
        
def calc_chem_properties_mol(mol):
    clogp = MolLogP(mol)
    mw = CalcExactMolWt(mol)
    tpsa = TPSA(mol)
    qed = QED.qed(mol)  # quantitative estimation of drug-likeness: ranges between 0 and 1, with 1 being the most drug-like.
    return mw, clogp, tpsa, qed


def calc_chem_properties(smi):
    try:
        mol = MolFromSmiles(smi)
        clogp = MolLogP(mol)
        mw = CalcExactMolWt(mol)
        tpsa = TPSA(mol)
        qed = QED.qed(mol)  # quantitative estimation of drug-likeness: ranges between 0 and 1, with 1 being the most drug-like.
    except:
        mw = 0.
        clogp = 0.
        tpsa = 0.
        qed = 0.
    return mw, clogp, tpsa, qed
    
    
def calc_structural_similarity(smi_1, smi_2):
    try:
        mol_1 = MolFromSmiles(smi_1)
        mol_2 = MolFromSmiles(smi_2)
        fp_1 = GetMorganFingerprintAsBitVect(mol_1, 2, nBits=2048, useChirality=False)
        fp_2 = GetMorganFingerprintAsBitVect(mol_2, 2, nBits=2048, useChirality=False)
        score = TanimotoSimilarity(fp_1, fp_2)
    except:
        score = 0.
    return score