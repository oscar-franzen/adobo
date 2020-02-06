# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for bulk RNA-seq integration
"""
import numpy as np
import pandas as pd
from .normalize import clean_matrix

def _deconv_basis(counts_sc, ct, samples):
    ids = ct.to_numpy() + '_' + samples.to_numpy()
    n = counts_sc.to_numpy()
    
    mean_mat = []
    for id in np.unique(ids):
        sub = counts_sc.iloc[:, ids==id]
        s = sub.sum().sum()
        mean_mat.append(np.apply_along_axis(lambda y: sum(y)/s, 1, sub))
    mean_mat = pd.DataFrame(mean_mat).transpose()
    mean_mat.index = counts_sc.index
    mean_mat.columns = np.unique(ids)
    
    arr = np.array(mean_mat.columns.str.split('_').tolist())[:, 0]
    sigma = []
    for ct_ in np.unique(arr):
        y = mean_mat.loc[:, arr==ct_]
        sigma.append(y.var(axis=1))
    sigma = pd.concat(sigma, axis=1)
    
    sum_mat2 = []
    for id in np.unique(samples):
        foo = []
        for ct_ in np.unique(ct):
            sub = n[:, np.logical_and(samples==id, ct==ct_)]
            foo.append(sub.sum()/sub.shape[1])
        sum_mat2.append(foo)
    sum_mat2 = pd.DataFrame(np.array(sum_mat2).T, index=np.unique(ct),
                            columns=np.unique(samples))
    sum_mat = sum_mat2.mean(axis=1)
    basis = []
    for ct_ in np.unique(arr):
        y = mean_mat*sum_mat[arr].values
        basis.append((y.loc[:, arr==ct_]).mean(axis=1))
    basis = pd.concat(basis, axis=1)
    basis.columns = np.unique(arr)
    
    var_adj = []
    for sid in np.unique(samples):
        v = []
        for ct_ in np.unique(ct):
            y = counts_sc.loc[:, np.logical_and(samples==sid, ct==ct_)]
            v.append(y.var(axis=1))
        v = pd.concat(v, axis=1)
        x = v.max(axis=1)
        y = x/np.median(x)
        var_adj.append(y)
    var_adj = pd.concat(var_adj, axis=1)
    var_adj.columns = np.unique(samples)
    q15 = var_adj.quantile(q=0.15, axis=0)
    q85 = var_adj.quantile(q=0.85, axis=0)
    
    def adj(r):
        r[r<q15] = q15[r<q15]
        r[r>q85] = q85[r>q85]
        return r
    
    var_adj_q = np.apply_along_axis(adj, 1, var_adj.to_numpy())
    var_adj_q = pd.DataFrame(var_adj_q, index=var_adj.index,
                             columns=var_adj.columns)
    
    mean_mat_mvw = []
    for id in np.unique(ids):
        sid = id.split('_')[1]
        y = counts_sc.loc[:, samples==sid]
        yy = y.div(np.sqrt(var_adj_q[sid]), axis=0)
        yy[yy.isna()] = 0
        mean_mat_mvw.append(yy.sum(axis=1)/yy.sum().sum())
    mean_mat_mvw = pd.concat(mean_mat_mvw, axis=1)
    mean_mat_mvw.columns = np.unique(ids)
    
    basis_mvw = []
    for ct_ in np.unique(arr):
        y = mean_mat_mvw*sum_mat[arr].values
        basis_mvw.append((y.loc[:, arr==ct_]).mean(axis=1))
    basis_mvw = pd.concat(basis_mvw, axis=1)
    basis_mvw.columns = np.unique(arr)
    
    return basis, sum_mat, sigma, basis_mvw, var_adj_q

def deconv(obj, bulk, cell_type_var, sample_var=None, target_ct=None,
           verbose=False):
    """Implements deconvolution of bulk RNA-seq data into cell types
    similar to the method in Dong2019

    Notes
    -----
    Finds two non-negative matrices
        1) a basis matrix, B, dimensions N x K (N genes and K cell
           types)
        2) a proportion matrix, P, dimensions K x M (K cell types and
           M samples)
    satisfying:
        Y = B x P
    where Y is a bulk gene expression matrix with dimensions N x M
    (N genes and M samples)

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    bulk : `pandas.DataFrame`
        A pandas data frame containing the bulk gene expression. Genes
        as rows and samples as columns.
    cell_type_var : `str`
        A string (column name) of the meta data variable that contains
        the cell types. Can be set to None or empty to instead use
        clusters.
    sample_var : `str`
        A string (column name) of the meta data variable that contains
        sample names. If this is None or empty, then it is assumed all
        cells come from the same sample.
    target_ct : `list`
        A list of cell types to use. Default: None
    verbose : `bool`
        Be verbose or not. Default: False

    Example
    -------
    # Add example

    References
    ----------
    .. [1] Dong et al. (2019) https://www.biorxiv.org/content/10.1101/743591v1?rss=1
           SCDC: Bulk Gene Expression Deconvolution by Multiple Single-Cell RNA
           Sequencing References
    .. [2] https://github.com/meichendong/SCDC

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if not isinstance(bulk, pd.DataFrame):
        raise ValueError('"bulk" should be a pandas data frame')
    counts_sc = obj.count_data.copy()
    if obj.sparse:
        counts_sc = counts_sc.sparse.to_dense()
    counts_sc, meta = clean_matrix(counts_sc,
                                   obj,
                                   remove_low_qual=True,
                                   remove_mito=True,
                                   meta=True)
    if target_ct != None:
        counts_sc = counts_sc.loc[:, ct.isin(target_ct)]
        meta = meta[meta[[cell_type_var]].iloc[:, 0].isin(target_ct)]
    if sample_var == None:
        samples = np.ones(len(counts_sc.columns))
    else:
        samples = meta[[sample_var]].iloc[:, 0]
    ct = meta[[cell_type_var]].iloc[:, 0]
    counts_sc = counts_sc[counts_sc.mean(axis=1)>0]
    bulk = bulk[bulk.mean(axis=1)>0]

    # take common genes
    counts_sc = counts_sc[counts_sc.index.isin(bulk.index)]
    bulk = bulk[bulk.index.isin(counts_sc.index)]
    counts_sc = counts_sc.reindex(bulk.index)
    
    # estimate basis matrix
    basis, sum_mat, sigma, basis_mvw, var_adj_q = _deconv_basis(counts_sc,
                                                                ct,
                                                                samples)
