import os
import pandas as pd
import tqdm
from rdkit import Chem


if __name__=="__main__":
    input_dir = 'raw'
    filenames = os.listdir(input_dir)
    
    ## https://www.tcichemicals.com/KR/en/support-download/chemistry-clip/2013-07-02#:~:text=The%20biologically%20essential%20metal%20elements,being%20used%20for%20pharmaceutical%20purposes.
    atomicset = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', 'P'}
    
    frames = []
    for filename in filenames:
        df = pd.read_csv(os.path.join(input_dir, filename), sep='\t')
        print(f'[{filename}] : {df.shape}')
        frames.append(df)
        
    df_merged = pd.concat(frames, ignore_index=True)
    print(df_merged.shape)
    print(df_merged.head())
        
    records = []
    for i in tqdm.trange(df_merged.shape[0]):
        lnt = 0
        atm = False
        idx = df_merged.loc[i, 'zinc_id']
        mwt = df_merged.loc[i, 'mwt']
        lgp = df_merged.loc[i, 'logp']
        try:
            mol = Chem.MolFromSmiles(df_merged.loc[i, 'smiles'])
            ## length
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            lnt = len(smi)
            ## atoms
            atm = {atom.GetSymbol() for atom in mol.GetAtoms()}.issubset(atomicset)
        except:
            pass
        ## check
        if 0 < lnt <= 150 and atm:
            records.append((idx, smi, mwt, lgp, lnt))
    
    df_res = pd.DataFrame.from_records(records)
    df_res = df_res.rename(columns={0:'zinc_id', 1:'smiles', 2:'mwt', 3:'logp', 4:'length'})
    
    df_res.to_csv('zinc15_raw_to_canonical.csv', sep=',', index=False)
