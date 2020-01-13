# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
#     Contact: Oscar Franzén <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for reading and writing scRNA-seq data.
"""
import re
import os
import time
import subprocess

import datatable as dt
import pandas as pd
import numpy as np

import adobo._log
from .data import dataset


def export_data(obj, filename, norm='standard', clust='leiden',
                what='normalized', transpose=False, sep='\t',
                row_names=True, min_cluster_size=10,
                genes_uppercase=False):
    """Exports data to a text file, convenient for loading into other
    programs

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    filename : `str`
        Output filename or path.
    norm : `str`
        Name of the normalisation. For example 'standard'.
    clust : `str`
        Name of the clustering. For example: 'leiden̈́'.
    what : `{'normalized', 'clusters', 'pca', 'tsne', 'umap', 'cell_type_pred',
             'median_expr'}`
        What to export.
    transpose : `bool`
        Transpose the data before writing it. Default: False
    sep : `str`
        A character or regular expression used to separate
        fields. Default: "\t"
    row_names : `bool`
        Write row names or not. Default: True
    min_cluster_size : `int`
        Minimum number of cells per cluster; clusters smaller than
        this are ignored.  Default: 10
    genes_uppercase : `bool`
        Transform gene symbols to uppercase. Default: False

    Returns
    -------
    Nothing.
    """
    choices = ('normalized', 'clusters', 'pca',
               'tsne', 'umap', 'cell_type_pred', 'median_expr')
    if not what in choices:
        raise Exception(
            '"what" must be one of: %s' % ', '.join(choices))
    if what == 'normalized':
        if obj.sparse:
            # or else to_csv takes tpp long time
            D = obj.norm_data[norm]['data'].sparse.to_dense()
        else:
            D = obj.norm_data[norm]['data']
    elif what == 'pca':
        D = obj.norm_data[norm]['dr']['pca']['comp']
    elif what == 'clusters':
        D = pd.DataFrame(obj.norm_data[norm]['clusters'][clust]['membership'])
        D.columns = [clust]
    elif what == 'tsne':
        D = obj.norm_data[norm]['dr']['tsne']['embedding']
    elif what == 'umap':
        D = obj.norm_data[norm]['dr']['umap']['embedding']
    elif what == 'cell_type_pred':
        D = obj.norm_data[norm]['clusters'][clust]['cell_type_prediction']
    elif what == 'median_expr':
        cl = obj.norm_data[norm]['clusters'][clust]['membership']
        D = obj.norm_data[norm]['data']
        ret = D.groupby(cl.values, axis=1).aggregate(np.median)
        q = pd.Series(cl).value_counts()
        cl_remove = q[q < min_cluster_size].index
        ret = ret.iloc[:, np.logical_not(ret.columns.isin(cl_remove))]
        D = ret
    if genes_uppercase:
        D.index = D.index.str.upper()
    if transpose:
        D = D.transpose()
    D.to_csv(filename, sep=sep, index=row_names)

def reader(filename, sep='\s', header=True, do_round=False, **args):
    """Load a gene expression matrix from a file

    Parameters
    ----------
    filename : `str`
        Path to the file containing input data. Should be a matrix
        where columns are cells and rows are genes.
    sep : `str`
        A character or regular expression used to separate
        fields. Default: "\\s" (i.e. any white space character)
    header : `bool`
        If the data file has a header or not. Default: True
    do_round : `bool`
        In case of read count fractions, round to integers. Can be a
        useful remedy if read counts have been imputed or
        similar. Default: False

    Returns
    -------
    :class:`pandas.DataFrame`
        A data frame.
    """
    skip_to_line = 1
    if header:
        skip_to_line = 2
    count_data = dt.fread(
        filename, skip_to_line=skip_to_line, **args).to_pandas()
    count_data.index = count_data.iloc[:, 0]
    count_data = count_data.drop(count_data.columns[0], axis=1)
    if header:
        tool = 'cat'
        if re.search('.gz$', filename):
            tool = 'zcat'
        elif re.search('.zip$', filename):
            tool = 'unzip -c'
        elif re.search('.bz2$', filename):
            tool = 'bzcat'
        elif re.search('.xz$', filename):
            tool = 'xzcat'
        cmd = '%s %s | head -n1' % (tool, filename)
        h = subprocess.check_output(cmd, shell=True).decode(
            'ascii').replace('\n', '')
        if sep == '\s':
            hs = re.split('[\s,]', h)
            if len(hs) > 1:
                if len(hs) == count_data.shape[1]:
                    count_data.columns = hs
                else:
                    count_data.columns = hs[1:len(hs)]
            else:
                if verbose:
                    print('Skipping to set columns (mismatch in \
length for header).')
    # remove duplicate genes
    dups = count_data.index.duplicated(False)
    if np.any(dups):
        count_data = count_data.iloc[np.logical_not(dups)]
        if verbose:
            print('%s duplicated genes detected and removed.' % np.sum(dups))
    t = count_data.dtypes.unique()[0]
    if t != np.int32 and t != np.int64:
        if do_round:
            count_data = count_data.astype(int)
        else:
            raise Exception('Non-count values detected in data matrix, consider \
setting do_round=True, but first of all make sure your input data are raw read \
counts and not normalized counts.')
    rem = count_data.index.str.contains(
        '^ArrayControl-[0-9]+', regex=True, case=False)
    count_data = count_data[np.logical_not(rem)]
    count_data.index = count_data.index.str.replace('"', '')
    count_data.columns = count_data.columns.str.replace('"', '')
    return count_data


def load_from_file(filename, sep='\s', header=True, desc='no desc set',
                   output_file=None, sparse=True, bundled=False,
                   do_round=False, verbose=False, **args):
    r"""Load a gene expression matrix consisting of raw read counts

    Notes
    -----
    The loaded gene expression matrix should not have been
    normalized. This function calls :func:`~datatable.fread` to read
    the data matrix file. Any additional arguments are passed into it.

    Parameters
    ----------
    filename : `str`
        Path to the file containing input data. Should be a matrix
        where columns are cells and rows are genes.
    sep : `str`
        A character or regular expression used to separate
        fields. Default: "\\s" (i.e. any white space character)
    header : `bool`
        If the data file has a header or not. Default: True
    desc : `str`
        A description of the data
    output_file : `str`
        An output filename used when calling
        :py:func:`adobo.data.dataset.save()`.
    sparse : `bool`
        Represent the data in a sparse data structure. Will save
        memory at the expense of time. Default: True
    bundled : `bool`
        Use data installed by adobo. Default: False
    do_round : `bool`
        In case of read count fractions, round to integers. Can be a
        useful remedy if read counts have been imputed or
        similar. Default: False
    verbose : `bool`
        To be verbose or not. Default: False

    Returns
    -------
    :class:`adobo.data.dataset`
        A dataset class object.
    """
    if bundled:
        if re.search('/', filename):
            raise Exception(
                'If bundled=True, just specify a file name, not a path.')
        filename = '/'.join(adobo._log.__file__.split('/')
                            [0:-1]) + '/data/' + filename
    if not os.path.exists(filename):
        raise Exception('%s not found' % filename)
    stime = time.time()
    #count_data = pd.read_csv(filename, delimiter=sep, header=header, **args)
    count_data = reader(filename, sep, header, do_round, **args)
    obj = dataset(count_data, desc, output_file=output_file,
                  input_file=filename, sparse=sparse, verbose=verbose)
    if verbose:
        genes = '{:,}'.format(count_data.shape[0])
        cells = '{:,}'.format(count_data.shape[1])
        print('%s genes and %s cells were loaded' % (genes, cells))
        etime = time.time()
        print('loading took %.1f minutes' % ((etime-stime)/60))
    return obj
