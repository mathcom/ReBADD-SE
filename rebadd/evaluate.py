from rdkit import Chem, RDLogger
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
RDLogger.DisableLog('rdApp.*')


def evaluate_sr_nov_div(df, ref, objs, drop_invalid=True, drop_duplicates=True):
    if objs not in OBJECTIVES:
        print(f'[ERROR] acceptable objs: {OBJECTIVES.keys()}')
        return (0., 0., 0.)
    
    calc_sr = OBJECTIVES.get(objs)
    
    ## Validity
    if drop_invalid:
        idx = df.loc[:,'smiles'].map(lambda x:Chem.MolFromSmiles(x) is not None)
        df = df[idx]
    
    ## Success Rate
    succeeded, s_sr = calc_sr(df)
    if drop_duplicates:
        succeeded = list(set(succeeded))
    
    ## Novelty
    s_nov = calc_nov(succeeded, ref)
    
    ## Diversity
    s_div = calc_div(succeeded)
    
    return (s_sr, s_nov, s_div)


class TanimotoSimilarity_OneToBulk:
    def __init__(self, bs, aggregate=None):
        self.bs = bs
        self.b_fps = [self._fingerprints_from_smi(smi) for smi in self.bs]
        self.aggregate = aggregate
        
    def __call__(self, a):
        a_fp = self._fingerprints_from_smi(a)
        outputs = BulkTanimotoSimilarity(a_fp, self.b_fps)
        if self.aggregate == 'max':
            return max(outputs)
        elif self.aggregate == 'sum':
            return sum(outputs)
        else:
            return outputs
        
    def _fingerprints_from_smi(self, smi):
        mol = Chem.MolFromSmiles(smi)
        fp = GetMorganFingerprintAsBitVect(mol, 3, nBits=2048, useChirality=False)
        return fp
    
    
def calc_nov(succeeded, refecnce):
    
    if len(succeeded) > 0:
        calc_sim = TanimotoSimilarity_OneToBulk(refecnce, aggregate='max')

        s_nov = 0

        for smi in succeeded:
            s_nov += 1 if calc_sim(smi) < 0.4 else 0

        s_nov = s_nov / len(succeeded)
        
        return s_nov
    else:
        return 0.

    
def calc_div(succeeded):
    
    if len(succeeded) > 1:
        calc_sim = TanimotoSimilarity_OneToBulk(succeeded, aggregate='sum')

        s_div = -len(succeeded) # to consider self-similarities (i.e. (x_i,x_j) where i=j)

        for smi in succeeded:
            s_div += calc_sim(smi)

        s_div = 1. - s_div / (len(succeeded) * (len(succeeded)-1))

        return s_div
    else:
        return 0.
    
    
def calc_sr_gsk3b_jnk3_qed_sq(df):
    succeeded = []
    
    for smi, s_gsk, s_jnk, s_qed, s_sa in df.loc[:,('smiles', 'gsk3b', 'jnk3', 'qed', 'sa')].values:
        is_succeed = True
        is_succeed &= s_gsk > 0.5 - EPSILON
        is_succeed &= s_jnk > 0.5 - EPSILON
        is_succeed &= s_qed > 0.6 - EPSILON
        is_succeed &= s_sa < 4.0 + EPSILON
        
        if is_succeed:
            succeeded.append(smi)
        
    s_sr = len(succeeded) / len(df)
        
    return succeeded, s_sr
    
    
def calc_sr_gsk3b_jnk3(df):
    succeeded = []
    
    for smi, s_gsk, s_jnk in df.loc[:,('smiles', 'gsk3b', 'jnk3')].values:
        is_succeed = True
        is_succeed &= s_gsk > 0.5 - EPSILON
        is_succeed &= s_jnk > 0.5 - EPSILON
        
        if is_succeed:
            succeeded.append(smi)
        
    s_sr = len(succeeded) / len(df)
        
    return succeeded, s_sr


def calc_sr_gsk3b(df):
    succeeded = []
    
    for smi, s_gsk in df.loc[:,('smiles', 'gsk3b')].values:
        is_succeed = True
        is_succeed &= s_gsk > 0.5 - EPSILON
        
        if is_succeed:
            succeeded.append(smi)
        
    s_sr = len(succeeded) / len(df)
        
    return succeeded, s_sr


def calc_sr_jnk3(df):
    succeeded = []
    
    for smi, s_jnk in df.loc[:,('smiles', 'jnk3')].values:
        is_succeed = True
        is_succeed &= s_jnk > 0.5 - EPSILON
        
        if is_succeed:
            succeeded.append(smi)
        
    s_sr = len(succeeded) / len(df)
        
    return succeeded, s_sr
    
    
    
def calc_sr_bcl2_bclxl_bclw(df):
    '''
    average criteria
    '''
    succeeded = []
    
    for smi, s_bcl2, s_bclxl, s_bclw in df.loc[:,('smiles', 'bcl2', 'bclxl', 'bclw')].values:
        is_succeed = True
        is_succeed &= (s_bcl2 > 9.069 - EPSILON)
        is_succeed &= (s_bclxl > 8.283 - EPSILON)
        is_succeed &= (s_bclw > 6.999 - EPSILON)
        if is_succeed:
            succeeded.append(smi)
        
    s_sr = len(succeeded) / len(df)
        
    return succeeded, s_sr
    
    
EPSILON = 1e-6

OBJECTIVES = {
    'gsk3b_jnk3_qed_sa':calc_sr_gsk3b_jnk3_qed_sq,
    'gsk3b_jnk3':calc_sr_gsk3b_jnk3,
    'gsk3b':calc_sr_gsk3b,
    'jnk3':calc_sr_jnk3,
    'bcl2_bclxl_bclw':calc_sr_bcl2_bclxl_bclw,
}