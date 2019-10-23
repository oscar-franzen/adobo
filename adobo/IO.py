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
import time

import datatable as dt
import pandas as pd
import numpy as np

from adobo import dataset

def load_from_file(filename, sep='\s', header=0, desc='no desc set', output_file=None,
                   sparse=True, verbose=False, **args):
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
    :func:`~datatable.fread` to read the data matrix file. Any additional arguments are
    passed into it.

    Returns
    -------
    :class:`adobo.data.dataset`
        A dataset class object.
    """
    if not os.path.exists(filename):
        raise Exception('%s not found' % filename)
    stime = time.time()
    #count_data = pd.read_csv(filename, delimiter=sep, header=header, **args)
    count_data = dt.fread(filename, **args).to_pandas()
    count_data.index = count_data.iloc[:, 0]
    count_data = count_data.drop(count_data.columns[0], axis=1)
    # remove duplicate genes
    dups = count_data.index.duplicated(False)
    if np.any(dups):
        count_data = count_data.iloc[np.logical_not(dups)]
        if verbose:
            print('%s duplicated genes detected and removed.' % np.sum(dups))
    t = count_data.dtypes.unique()[0]
    if t != np.int32 and t != np.int64:
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
        etime = time.time()
        print('loading took %.1f minutes' % (etime-stime)/60)
    return obj
