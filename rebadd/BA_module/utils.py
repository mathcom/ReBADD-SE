import numpy as np
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

    
def get_regression_result(labels, predictions):
    labels = np.array(labels)
    predictions = np.array(predictions)

    RMSE = mean_squared_error(labels, predictions)**0.5
    PCC = pearsonr(labels, predictions)[0] # [0]: r, [1]: p-value
    CI = concordance_index(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    return RMSE, PCC, CI, r2
    