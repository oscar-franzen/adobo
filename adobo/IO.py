# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for reading and writing scRNA-seq data.
"""
import os
import pandas as pd
import numpy as np

from adobo import dataset

def load_from_file(filename, sep='\t', header=0, column_id='auto', verbose=False, **args):
    r"""Load a gene expression matrix consisting of raw read counts

    Parameters
    ----------
    filename : `str`
        Path to the file containing input data. Should be a matrix where
        columns are cells and rows are genes.
    sep : `str`, optional
        Character used to separate fields (default: "\\t").
    header : `str`, optional
        If the data file has a header. 0 means yes otherwise None (default: `0`).
    column_id : {'auto', 'yes', 'no'}, optional
        Whether the header (first line) of the file contains a column ID for the genes. If
        this is the case, set this to auto or yes, otherwise no (default: auto).
    verbose : `bool`, optional
        To be verbose or not (default: False).

    Notes
    -----
    The loaded gene expression matrix should not have been normalized. This function calls
    :func:`~pandas.io.parsers.read_csv` to read the data matrix file. Any additional
    arguments are passed into :func:`~pandas.io.parsers.read_csv`.

    Returns
    -------
    :class:`adobo.data.dataset`
        A dataset class object.
    """
    if not os.path.exists(filename):
        raise Exception('%s not found' % filename)
    if len(sep)>1:
        raise Exception('`sep` cannot be longer than 1, it should specify a single \
character.')
    if not column_id in ('auto', 'yes', 'no'):
        raise Exception('"column_id" can only be set to "auto", "yes" or "no"')
    exp_mat = pd.read_csv(filename,
                          delimiter=sep,
                          header=header,
                          **args)
    def move_col(x):
        x.index = x[x.columns[0]]
        x = x.drop(x.columns[0], axis=1)
        return x

    if column_id == 'auto':
        if exp_mat[exp_mat.columns[0]].dtype != int:
            exp_mat = move_col(exp_mat)
    elif column_id == 'yes':
        exp_mat = move_col(exp_mat)
            
    # remove duplicate genes
    dups = exp_mat.index.duplicated(False)
    if np.any(dups):
        exp_mat = exp_mat.iloc[np.logical_not(dups)]
        if verbose:
            print('%s duplicated genes detected and removed.' % np.sum(dups))
    if np.any(exp_mat.dtypes != 'int64'):
        raise Exception('Non-count values detected in data matrix.')
    rem = exp_mat.index.str.contains('^ArrayControl-[0-9]+', regex=True, case=False)
    exp_mat = exp_mat[np.logical_not(rem)]
    obj = dataset(exp_mat)
    if verbose:
        genes = '{:,}'.format(exp_mat.shape[0])
        cells = '{:,}'.format(exp_mat.shape[1])
        print('%s genes and %s cells were loaded' % (genes, cells))
    return obj
