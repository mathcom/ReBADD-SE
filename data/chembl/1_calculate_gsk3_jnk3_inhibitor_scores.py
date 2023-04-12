import os
import sys
import pandas as pd
import networkx as nx
import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'rebadd'))
if ROOT_PATH not in sys.path:
    sys.path = [ROOT_PATH] + sys.path
    
from chemutils import GSKScorer, JNKScorer, SAScorer, QEDScorer


def GetNumRings(mol):
    '''
    Reference: https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/properties.py
    '''
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    return len(cycle_list)


if __name__=='__main__':
    df_chembl = pd.read_csv('all_processed.csv')

    clf_gsk3 = GSKScorer()
    clf_jnk3 = JNKScorer()
    clf_sa = SAScorer()
    clf_qed = QEDScorer()

    scores_gsk3 = []
    scores_jnk3 = []
    scores_sa = []
    scores_qed = []
    for i in tqdm.trange(len(df_chembl)):
        smi = df_chembl.loc[i,"smiles"]
        mol = Chem.MolFromSmiles(smi)
        ## target properties
        df_chembl.loc[i,'gsk3'] = clf_gsk3(smi)
        df_chembl.loc[i,'jnk3'] = clf_jnk3(smi)
        df_chembl.loc[i,'sa'] = clf_sa(smi)
        df_chembl.loc[i,'qed'] = clf_qed(smi)
        ## number of atoms
        df_chembl.loc[i,'num_atoms'] = mol.GetNumAtoms()
        ## number of rings
        df_chembl.loc[i,'num_rings'] = GetNumRings(mol)
        
    df_chembl.to_csv('chembl_gsk3_jnk3_qed_sa.csv', index=False)