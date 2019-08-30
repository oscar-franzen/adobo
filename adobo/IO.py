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

def load_from_file(filename, sep='\t', header=0, column_id=True, verbose=False, **args):
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
    column_id : `bool`, optional
        Whether the header (first line) of the file contains a column ID for the genes. If
        this is the case, set this to True, otherwise False (default: True).
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
    exp_mat = pd.read_csv(filename,
                          delimiter=sep,
                          header=header,
                          **args)
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
    obj = dataset(exp_mat)
    if verbose:
        genes = '{:,}'.format(exp_mat.shape[0])
        cells = '{:,}'.format(exp_mat.shape[1])
        print('%s genes and %s cells were loaded' % (genes, cells))
    return obj
