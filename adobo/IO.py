# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzén <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for reading and writing scRNA-seq data.
"""
import os
import pandas as pd
import numpy as np

from adobo import dataset

def load_from_file(filename, sep='\s', header=0, column_id='auto', verbose=False,
                   desc='no desc set', output_file=None, input_file=None, sparse=True,
                   **args):
    r"""Load a gene expression matrix consisting of raw read counts

    Parameters
    ----------
    filename : `str`
        Path to the file containing input data. Should be a matrix where
        columns are cells and rows are genes.
    sep : `str`
        A character or regular expression used to separate fields. Default: "\\s"
        (i.e. any white space character)
    header : `str`
        If the data file has a header. 0 means yes otherwise None. Default: 0
    column_id : {'auto', 'yes', 'no'}
        Whether the header (first line) of the file contains a column ID for the genes. If
        this is the case, set this to auto or yes, otherwise no. Default: 'auto'
    desc : `str`
        A description of the data
    output_file : `str`
        An output filename used when calling :py:func:`adobo.data.dataset.save()`.
    sparse : `bool`
        Represent the data in a sparse data structure. Will save memory at the expense
        of time. Default: True
    verbose : `bool`
        To be verbose or not. Default: False

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
    if not column_id in ('auto', 'yes', 'no'):
        raise Exception('"column_id" can only be set to "auto", "yes" or "no"')
    count_data = pd.read_csv(filename,
                          delimiter=sep,
                          header=header,
                          **args)
    def move_col(x):
        x.index = x[x.columns[0]]
        x = x.drop(x.columns[0], axis=1)
        return x

    if column_id == 'auto':
        if count_data[count_data.columns[0]].dtype != int:
            count_data = move_col(count_data)
    elif column_id == 'yes':
        count_data = move_col(count_data)
            
    # remove duplicate genes
    dups = count_data.index.duplicated(False)
    if np.any(dups):
        count_data = count_data.iloc[np.logical_not(dups)]
        if verbose:
            print('%s duplicated genes detected and removed.' % np.sum(dups))
    if np.any(count_data.dtypes != 'int64'):
        raise Exception('Non-count values detected in data matrix.')
    rem = count_data.index.str.contains('^ArrayControl-[0-9]+', regex=True, case=False)
    count_data = count_data[np.logical_not(rem)]
    count_data.index = count_data.index.str.replace('"', '')
    count_data.columns = count_data.columns.str.replace('"', '')
    obj = dataset(count_data, desc, output_file=output_file, input_file=filename,
                  sparse=sparse, verbose=verbose)
    if verbose:
        genes = '{:,}'.format(count_data.shape[0])
        cells = '{:,}'.format(count_data.shape[1])
        print('%s genes and %s cells were loaded' % (genes, cells))
    return obj
