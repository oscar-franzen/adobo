# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for differential expression.
"""
import scipy.linalg
import statsmodels.api as sm
import patsy

import pandas as pd
import numpy as np

from ._stats import p_adjust_bh

def combine_tests(obj, normalization=(), clustering=(), method='simes',
                  min_cluster_size=10):
    """Uses Simes' method for combining p-values

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    normalization : `tuple`
        A tuple of normalization to use. If it has the length zero, then all available
        normalizations will be used.
    clustering : `tuple`, optional
        Specifies the clustering outcomes to work on.
    method : `{'simes'}`
        Method for combining p-values. Only Simes' is implemented.
    min_cluster_size : `int`
        Minimum number of cells per cluster (clusters smaller than this are ignored).
        Default: 10

    References
    ----------
    .. [1] Simes, R. J. (1986). An improved Bonferroni procedure for multiple tests of
           significance. Biometrika, 73(3):751-754. 

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if not method in ('simes',):
        raise Exception('Unsupported method for combining p-values. Only Simes method \
is available at the moment.')
    targets = {}
    if len(normalization) == 0 or normalization == '':
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    for i, k in enumerate(targets):
        item = targets[k]
        clusters = item['clusters']
        for algo in clusters:
            if len(clustering) == 0 or algo in clustering:
                try:
                    pval_mat = obj.norm_data[k]['de'][algo]['mat_format']
                except KeyError:
                    raise Exception('P-values have not been generated yet. Please run \
adobo.de.linear_model(...) first.')
                cl = clusters[algo]['membership']
                # remove clusters with too few cells
                q = pd.Series(cl).value_counts()
                res = []
                for cc in q[q >= min_cluster_size].index:
                    idx = pval_mat.columns.str.match('^%s_vs'%cc)
                    subset_mat = pval_mat.iloc[:,idx]
                    r = subset_mat.rank(axis=1)
                    T = (subset_mat.shape[1]*subset_mat/r).min(axis=1).sort_values()
                    T[T>1] = 1
                    df = pd.DataFrame({'cluster' : [cc]*len(T), 'gene' : T.index, 'pvalue.Simes' : T})
                    res.append(df)
                res = pd.concat(res)
                obj.norm_data[k]['de'][algo]['combined'] = res

def _choose_leftright_pvalues(left, right, direction):
    """Internal helper function."""
    if direction == 'up':
        pv = right
    elif direction == 'down':
        pv = left
    else:
        pv = np.minimum(left,right)*2
    return pv

def linear_model(obj, normalization=(), clustering=(), direction='up',
                 min_cluster_size=10, verbose=False):
    """Performs differential expression analysis between clusters using a linear model
    and t-statistics

    Notes
    -----
    Finds differentially expressed (DE) genes between clusters by fitting a linear model
    (LM) to gene expression (response variables) and clusters (explanatory variables) and
    performs pairwise t-tests for significance. Model coefficients are estimated via the
    ordinary least squares method and p-values are calculated using t-statistics. One
    benefit of using LM for DE is that computations are vectorized and therefore very fast.
    
    The ideas behind using LM to explore DE have been extensively covered in the
    limma R package.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    normalization : `tuple`
        A tuple of normalization to use. If it has the length zero, then all available
        normalizations will be used.
    clustering : `tuple`, optional
        Specifies the clustering outcomes to work on.
    direction : `{'up', 'down', 'any'}`
        Can be 'any' for any direction 'up' for up-regulated and 'down' for
        down-regulated. Normally we want to find genes being upregulated in cluster A
        compared with cluster B. Default: up
    min_cluster_size : `int`
        Minimum number of cells per cluster (clusters smaller than this are ignored).
        Default: 10
    verbose : `bool`
        Be verbose or not. Default: False

    References
    ----------
    .. [1] https://www.bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf
    .. [2] https://newonlinecourses.science.psu.edu/stat555/node/12/

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    targets = {}
    if len(normalization) == 0 or normalization == '':
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    for i, k in enumerate(targets):
        if verbose:
            print('Running differential expression analysis on prediction on the %s \
normalization' % k)
        item = targets[k]
        X = item['data'].transpose()
        clusters = item['clusters']
        for algo in clusters:
            if len(clustering) == 0 or algo in clustering:
                cl = clusters[algo]['membership']
                # remove clusters with too few cells
                q = pd.Series(cl).value_counts()
                cl_remove = q[q < min_cluster_size].index
                X_f = X.loc[np.logical_not(cl.isin(cl_remove)).values, :]
                cl = cl[np.logical_not(cl.isin(cl_remove))]
                
                # full design matrix
                dm_full = patsy.dmatrix('~ 0 + C(cl)', pd.DataFrame({'cl' : cl}))
                resid_df = dm_full.shape[0] - dm_full.shape[1]
                
                # gene expression should be the response
                lm = sm.regression.linear_model.OLS(endog=X_f, # response
                                                    exog=dm_full)
                res = lm.fit()
                coef = res.params # coefficients

                # computing standard errors
                # https://stats.stackexchange.com/questions/44838/how-are-the-standard-errors-of-coefficients-calculated-in-a-regression
                # http://web.mit.edu/~r/current/arch/i386_linux26/lib/R/library/limma/html/lm.series.html
                # residual variance for each gene
                dm_nrow = dm_full.shape[0]
                dm_ncol = dm_full.shape[1]
                sigma2 = ((X_f-dm_full.dot(coef))**2).sum(axis=0)/(dm_nrow-dm_ncol)
                
                q = dm_full.transpose().dot(dm_full)
                chol = np.linalg.cholesky(q)
                chol2inv = scipy.linalg.cho_solve((chol, False), np.eye(chol.shape[0]))
                std_dev = np.sqrt(np.diag(chol2inv))
                clusts = np.unique(cl)
                
                # mean gene expression for every gene in every cluster
                mge = [X_f.loc[(cl == o).values, :].mean() for o in clusts]

                # perform all pairwise comparisons of clusters (t-tests)
                comparisons = []
                out_t_stats = []
                out_pv = []
                out_lfc = []
                out_mge_g1 = []
                out_mge_g2 = []
                
                for kk, _ in enumerate(clusts):
                    ref_coef = coef.iloc[kk, :]
                    # recompute coefficients for contrasts
                    # https://genomicsclass.github.io/book/pages/interactions_and_contrasts.html
                    con = np.zeros((coef.shape[0], kk))
                    np.fill_diagonal(con, -1)
                    con[kk,] = 1
                    std_new = np.sqrt((std_dev**2).dot(con**2))
                    
                    for ii in np.arange(kk):
                        std_err = std_new[ii]**2*sigma2
                        target_cl = clusts[ii]
                        # log2 fold change, reminder: log2(A/B)=log2(A)-log2(B)
                        cur_lfc = ref_coef - coef.iloc[ii, :]
                        cur_lfc.index = std_err.index
                        # compute p-values
                        cur_t = cur_lfc/np.sqrt(std_err)
                        t_dist = scipy.stats.t(resid_df)
                        left = t_dist.cdf(cur_t)
                        right = t_dist.sf(cur_t)
                        pv1 = _choose_leftright_pvalues(left, right, direction)
                        pv2 = _choose_leftright_pvalues(right, left, direction)
                        
                        comparisons.append('%s_vs_%s' % (clusts[kk], clusts[ii]))
                        comparisons.append('%s_vs_%s' % (clusts[ii], clusts[kk]))
                        
                        out_pv.append(pd.Series(pv1))
                        out_pv.append(pd.Series(pv2))
                        
                        out_t_stats.append(pd.Series(cur_t))
                        out_t_stats.append(pd.Series(cur_t))
                        
                        out_lfc.append(pd.Series(cur_lfc))
                        out_lfc.append(pd.Series(cur_lfc*-1))
                        
                        out_mge_g1.append(mge[kk])
                        out_mge_g2.append(mge[ii])
                        
                        out_mge_g1.append(mge[ii])
                        out_mge_g2.append(mge[kk])

                out_merged = pd.concat(out_pv, axis=1)
                out_merged.columns = comparisons
                out_merged.index = X_f.columns
                pval = pd.concat(out_pv, ignore_index=True)
                lab1 = []
                lab2 = []
                
                for q in comparisons:
                    lab1.append(pd.Series([q]*X_f.columns.shape[0]))
                    lab2.append(pd.Series(X_f.columns))
                    
                ll = pd.DataFrame({'comparison_A_vs_B' : pd.concat(lab1, ignore_index=True),
                           'gene' : pd.concat(lab2, ignore_index=True),
                           'p_val' : pval,
                           'FDR' : p_adjust_bh(pval),
                           't_stat' : pd.concat(out_t_stats, ignore_index=True),
                           'logFC' : pd.concat(out_lfc, ignore_index=True),
                           'mean.A' : pd.concat(out_mge_g1, ignore_index=True),
                           'mean.B' : pd.concat(out_mge_g2, ignore_index=True)})
                obj.norm_data[k]['de'][algo] = {'long_format' : ll, 'mat_format' : out_merged}
