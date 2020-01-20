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
import gzip
import time
import subprocess

import datatable as dt
import pandas as pd
import numpy as np

from scipy.io import mmread

import adobo._log
from .data import dataset


def export_data(obj, filename, norm='standard', clust='leiden',
                what='normalized', transpose=False, sep='\t',
                row_names=True, min_cluster_size=10,
                genes_uppercase=False, do_round=True,
                compression=False):
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
    do_round : `bool`
        Round normalized gene expression values to two
        decimals. Default: True
    compression : `bool`
        Compress output with gzip. Will append '.gz' to the
        filename. Default: False

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
        if do_round:
            D = round(D, 2)
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
    D.to_csv(filename + '.gz' if compression else filename,
             sep=sep,
             index=row_names,
             compression='gzip' if compression else None)

def reader(filename, sep='\s', header=True, do_round=False,
           verbose=False, **args):
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
    verbose : `bool`
        Be verbose or not. Default: False

    Returns
    -------
    :class:`pandas.DataFrame`
        A data frame.
    """
    skip_to_line = 1
    if header:
        skip_to_line = 2
    count_data = dt.fread(filename,
                          skip_to_line=skip_to_line,
                          **args).to_pandas()
    count_data.index = count_data.iloc[:, 0]
    count_data = count_data.drop(count_data.columns[0], axis=1)
    if np.any(count_data.dtypes == bool):
        count_data = count_data.astype('int32')
    if header:
        tool = 'cat'
        if re.search('.gz$', filename):
            tool = 'zcat'
        elif re.search('.zip$', filename):
            tool = 'unzip -p'
        elif re.search('.bz2$', filename):
            tool = 'bzcat'
        elif re.search('.xz$', filename):
            tool = 'xzcat'
        cmd = '%s "%s" | head -n1' % (tool, filename)
        h = subprocess.check_output(cmd, shell=True).decode(
            'ascii').replace('\n', '')
        if sep == '\s':
            pat = '[\s,]'
        else:
            pat = sep
        hs = re.split(pat, h)
        if len(hs) > 1:
            if len(hs) == count_data.shape[1]:
                count_data.columns = hs
            else:
                hs = hs[1:len(hs)]
                if hs[-1] == '':
                    hs = hs[0:len(hs)-1]
            if len(hs) == count_data.shape[1]:
                count_data.columns = hs
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
    if do_round:
        count_data = count_data.astype(int)
    for gene, r in count_data.iterrows():
        if np.any(r.apply(lambda x: not x.is_integer())):
            raise Exception('Non-count values detected in data matrix \
(in gene "%s"), consider setting do_round=True, but first of all make \
sure your input data are raw read counts and not normalized counts.' % gene)
    rem = count_data.index.str.contains('^ArrayControl-[0-9]+',
                                        regex=True,
                                        case=False)
    count_data = count_data[np.logical_not(rem)]
    count_data.index = count_data.index.str.replace('"', '')
    count_data.columns = count_data.columns.str.replace('"', '')
    return count_data

def load_matrix_market(filename):
    """Loads data in the matrix market format

    Parameters
    ----------
    filename : `str`
        Path to the file containing input data. See notes in
        :py:func:`adobo.IO.load_from_file`.

    References
    ----------
    .. [1] https://math.nist.gov/MatrixMarket/formats.html#MMformat

    Returns
    -------
    :class:`pandas.DataFrame`
        A gene expression data frame
    """
    if not os.path.exists(filename):
        raise Exception('%s not found' % filename)
    cmd = 'tar -C /tmp/ -zxvf %s' % filename
    h = subprocess.check_output(cmd, shell=True).decode('ascii').rstrip('\n')
    files = h.split('\n')
    err = 'Input file %s should contain exactly three files with the following \
file names: barcodes.tsv.gz, genes.tsv.gz, matrix.mtx.gz'
    if len(files) != 3:
        raise Exception(err)
    if not 'barcodes.tsv.gz' in files or \
       not 'genes.tsv.gz' in files or \
       not 'matrix.mtx.gz' in files:
       raise Exception(err)

    mtx_file = gzip.open('/tmp/matrix.mtx.gz', 'r')
    cell_mat = mmread(mtx_file)
    cell_mat = cell_mat.todense()

    name_file = gzip.open('/tmp/barcodes.tsv.gz', 'rt')
    cell_names = name_file.read().splitlines()
    cells = []
    
    for z in cell_names:
        z = z.replace('"', '').split('\t')
        if len(z) > 1:
            if z[0] != '':
                cells.append(z[1])
        else:
            cells.append(z[0])

    g_file = gzip.open('/tmp/genes.tsv.gz', 'rt')
    genes = g_file.read().splitlines()
    symb = []

    for g in genes:
        g = g.replace('"', '').split('\t')
        if len(g) > 1:
            if g[0] != '':
                symb.append(g[1])
        else:
            symb.append(g[0])
    
    m = pd.DataFrame(cell_mat, symb, cells)
    os.system('rm /tmp/barcodes.tsv.gz /tmp/genes.tsv.gz /tmp/matrix.mtx.gz')
    return m
    
def load_from_file(filename, sep='\s', header=True, desc='no desc set',
                   output_file=None, sparse=True, bundled=False,
                   do_round=False, flip_axes=False, verbose=False, **args):
    r"""Load a gene expression matrix consisting of raw read counts

    Notes
    -----
    The loaded gene expression matrix should not have been
    normalized. This function calls :func:`~datatable.fread` to read
    the data matrix file. Any additional arguments are passed into it.

    Matrix market format: MM is a common data format in NCBI's Gene
    Expression Omnibus. If the input file ends with "tar.gz", then it
    is assumed to be a gzip-compressed tar archive, containing data in
    the matrix market format. When extracting this file there should
    be exactly *three* files with exactly these file names: (i)
    matrix.mtx.gz, (ii) barcodes.tsv.gz, and (iii) genes.tsv.gz. See
    reference for more information about this format.

    Parameters
    ----------
    filename : `str`
        Path to the file containing input data. Should be a matrix
        where columns are cells and rows are genes. The input file can
        be compressed (gzip, bzip, zip, and xz are supported). The
        matrix market format is also supported (see notes).
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
    flip_axes : `bool`
        Rotate the data after loading it. Use if in the input data the
        genes are columns and cells are rows. Default: False
    verbose : `bool`
        To be verbose or not. Default: False

    References
    ----------
    .. [1] https://math.nist.gov/MatrixMarket/formats.html#MMformat

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
    if re.search('\.tar\.gz$', filename):
        if verbose:
            print('Input file name ends with "tar.gz", assuming matrix market\
 data.')
        count_data = load_matrix_market(filename)
    else:
            count_data = reader(filename, sep, header, do_round,
                                verbose, **args)
    if flip_axes:
        count_data = count_data.T
    obj = dataset(count_data, desc, output_file=output_file,
                  input_file=filename, sparse=sparse, verbose=verbose)
    if verbose:
        genes = '{:,}'.format(count_data.shape[0])
        cells = '{:,}'.format(count_data.shape[1])
        print('%s genes and %s cells were loaded' % (genes, cells))
        etime = time.time()
        print('loading took %.1f minutes' % ((etime-stime)/60))
    return obj
