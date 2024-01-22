import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pickle
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class jnk3_model():
    """Scores based on an ECFP classifier for activity."""

    clf_path = os.path.join(os.path.dirname(__file__), 'jnk3.pkl')

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, mol):
        fp = self.fingerprints_from_mol(mol)
        score = self.clf.predict_proba(fp)[:, 1]
        return float(score)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
