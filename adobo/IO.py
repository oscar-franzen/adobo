import os
import pandas as pd
import numpy as np

from adobo import data

def load_from_file(filename, sep='\t', header=0, column_id=True, verbose=False):
    """Load a gene expression matrix consisting of raw read counts

    Parameters
    ----------
        filename : str
                   Path to the file containing input data. Should be a matrix where
                   columns are cells and rows are genes.
        sep : str, optional
              Character used to separate fields (default \t)
        header : str, optional
                 If the data file has a header (default 0)

    Extended Summary
    ----------------
    Gene expression matrix should not be normalized.

    Returns
    -------
    A data class object.    
    """
    if not os.path.exists(filename):
        raise Exception('%s not found' % filename)
    exp_mat = pd.read_csv(filename,
                          delimiter=sep,
                          header=header)
    if column_id:
        exp_mat.index = exp_mat[exp_mat.columns[0]]
        exp_mat = exp_mat.drop(exp_mat.columns[0], axis=1)
    # remove duplicate genes
    dups = exp_mat.index.duplicated(False)
    if np.any(dups):
        exp_mat = exp_mat.iloc[np.logical_not(dups)]
        if verbose:
            print('%s duplicated genes detected and removed.' % np.sum(dups))
    if np.any(exp_mat.dtypes != 'int64'):
        raise Exception('Non-count values detected in data matrix.')
    obj = data(exp_mat)
    if verbose:
        genes = '{:,}'.format(exp_mat.shape[0])
        cells = '{:,}'.format(exp_mat.shape[1])
        print('%s genes and %s cells were loaded' % (genes, cells))
    return obj
