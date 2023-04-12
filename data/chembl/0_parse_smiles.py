import os
import pandas as pd
import tqdm
from rdkit import Chem


if __name__=="__main__":
    filename = 'all.txt'
    
    ## https://www.tcichemicals.com/KR/en/support-download/chemistry-clip/2013-07-02#:~:text=The%20biologically%20essential%20metal%20elements,being%20used%20for%20pharmaceutical%20purposes.
    atomicset = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', 'P'}
    
    df = pd.read_csv(filename, header=None).rename(columns={0:'smiles'})
    print(df.shape)
    print(df.head())
        
    records = []
    for i in tqdm.trange(df.shape[0]):
        lnt = 0
        atm = False
        try:
            mol = Chem.MolFromSmiles(df.loc[i, 'smiles'])
            ## length
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            lnt = len(smi)
            ## atoms
            atm = {atom.GetSymbol() for atom in mol.GetAtoms()}.issubset(atomicset)
        except:
            pass
        ## check
        if 0 < lnt <= 150 and atm:
            records.append((smi, lnt))
    
    df_res = pd.DataFrame.from_records(records)
    df_res = df_res.rename(columns={0:'smiles', 1:'length'})
    
    df_res.to_csv('all_processed.csv', sep=',', index=False)
