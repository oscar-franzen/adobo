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
import time
from multiprocessing import Pool
import psutil
import scipy.linalg
from scipy.stats import combine_pvalues as scipy_combine_pvalues
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import patsy

import pandas as pd
import numpy as np

from ._stats import p_adjust_bh


def filter(obj, normalization=None, clust_alg=None, thres=0.01, frac=0.8,
           retx=False):
    """Filters combined tests according to percent of cells expressing
    a gene

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    normalization : `str`
        The name of the normalization to operate on. If this is empty
        or None then the function will be applied on the last
        normalization that was applied.
    clust_alg : `str`
        Name of the clustering strategy. If empty or None, the last
        one will be used.
    thres : `float`
        Significance threshold for multiple testing
        correction. Default: 0.01
    frac : `float`
        Fraction of cells of the cluster that must express a
        gene. Default: 0.8
    retx : `bool`
        Returns a data frame with results (only modifying the object
        if False).  Default: True

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp)
    >>> ad.hvg.find_hvg(exp)
    >>> ad.dr.pca(exp)
    >>> ad.clustering.generate(exp)
    >>> ad.de.linear_model(exp)
    >>> ad.de.combine_tests(exp)
    >>> ad.de.filter(exp)

    Returns
    -------
    pandas.DataFrame or None (depending on `retx`)
        Differential expression results.
    """
    if normalization == None or normalization == '':
        norm = list(obj.norm_data.keys())[-1]
    else:
        norm = normalization
    try:
        target = obj.norm_data[norm]
    except KeyError:
        raise Exception('"%s" not found' % norm)
    if clust_alg == None or clust_alg == '':
        clust_alg = list(target['clusters'].keys())[-1]
    try:
        ct = target['de'][clust_alg]['combined'].copy()
    except KeyError:
        raise Exception('Run adobo.de.combine_tests(...) first.')
    ct = ct[np.logical_not(ct.mtc_p.isna())]
    ct = ct[ct.mtc_p < thres]
    cl = target['clusters'][clust_alg]['membership']
    X = target['data']
    res = []

    for c in np.unique(ct.cluster):
        X_ss = X.loc[:, cl == c]
        X_ss = X_ss.loc[ct[ct.cluster == c].gene, :]
        r = (X_ss > 0).sum(axis=1)
        g = r[r > X_ss.shape[1]*frac]
        z = ct[np.logical_and(ct.cluster == c, ct.gene.isin(g.index))]
        z['perc_cells'] = np.round((g.values/X_ss.shape[1])*100,2)
        res.append(z)
    res = pd.concat(res)
    res = res.sort_values(['cluster', 'combined_pvalue'],
                          ascending=[True, True])
    df = [[i, len(qwe.cluster), ' '.join(qwe.cluster.astype(str))] for i, qwe in res.groupby(by='gene')]
    df = pd.DataFrame(df)
    df.columns = ['gene', 'nclusters', 'clusters']
    df = df.sort_values(['nclusters'], ascending=[True])
    obj.norm_data[norm]['de'][clust_alg]['filtered'] = res
    obj.norm_data[norm]['de'][clust_alg]['summary'] = df
    if retx:
        return res


def combine_tests(obj, normalization=None, clust_alg=None, method='fisher',
                  min_cluster_size=10, mtc='BH', retx=False, verbose=False):
    """Generates a set of marker genes for every cluster by combining
    tests from pairwise analyses.

    Notes
    -----
    Run `adobo.de.linear_model` before running this function.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    normalization : `str`
        The name of the normalization to operate on. If this is empty
        or None then the function will be applied on the last
        normalization that was applied.
    clust_alg : `str`
        Name of the clustering strategy. If empty or None, the last
        one will be used.
    method : `{'fisher', 'simes', 'stouffer'}`
        Method for combining p-values. Default: fisher
    min_cluster_size : `int`
        Minimum number of cells per cluster (clusters smaller than
        this are ignored).  Default: 10
    mtc : `{'BH', 'bonferroni'}`
        Method to use for multiple testing correction. BH is
        Benjamini-Hochberg's procedure. Default: 'BH'
    retx : `bool`
        Returns a data frame with results (only modifying the object
        if False).  Default: False
    verbose : `bool`
        Be verbose or not. Default: False

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp)
    >>> ad.hvg.find_hvg(exp)
    >>> ad.dr.pca(exp)
    >>> ad.clustering.generate(exp)
    >>> ad.de.linear_model(exp)
    >>> ad.de.combine_tests(exp)

    References
    ----------
    .. [1] Simes, R. J. (1986). An improved Bonferroni procedure for
           multiple tests of significance. Biometrika, 73(3):751-754.
    .. [2] https://tinyurl.com/yxy3dy4v
    .. [3] https://en.wikipedia.org/wiki/Fisher%27s_method

    Returns
    -------
    pandas.DataFrame or None (depending on `retx`)
        Differential expression results.
    """
    if not method in ('simes', 'fisher', 'stouffer'):
        raise Exception('Unsupported method for combining p-values. Methods \
available: simes, fisher, and stouffer')
    if normalization == None or normalization == '':
        norm = list(obj.norm_data.keys())[-1]
    else:
        norm = normalization
    try:
        target = obj.norm_data[norm]
    except KeyError:
        raise Exception('"%s" not found' % norm)
    if clust_alg == None or clust_alg == '':
        clust_alg = list(target['clusters'].keys())[-1]
    cl = target['clusters'][clust_alg]['membership']
    try:
        pval_mat = obj.norm_data[norm]['de'][clust_alg]['mat_format']
    except KeyError:
        raise Exception('P-values have not been generated yet. Please first run \
adobo.de.linear_model or adobo.de.wilcox(...)')
    # remove clusters with too few cells
    q = pd.Series(cl).value_counts()
    res = []
    for cc in q[q >= min_cluster_size].index:
        if verbose:
            print('Working on cluster %s/%s' %
                  (cc, len(q[q >= min_cluster_size])-1))
        idx = pval_mat.columns.str.match('^%s_vs' % cc)
        subset_mat = pval_mat.iloc[:, idx]
        if method == 'simes':
            r = subset_mat.rank(axis=1)
            T = (subset_mat.shape[1]*subset_mat/r).min(axis=1).sort_values()
            T[T > 1] = 1
        else:
            T = []
            for gene, r in subset_mat.iterrows():
                if method == 'stouffer':
                    r[r == 0] = min(r[r > 0])  # a p=0 results in NaN
                r = r[r.notna()]
                T.append(scipy_combine_pvalues(r, method=method)[1])
            T = pd.Series(T, index=subset_mat.index)
        df = pd.DataFrame({'cluster': [cc]*len(T),
                           'gene': T.index, 'combined_pvalue': T})
        res.append(df)
    res = pd.concat(res)
    if mtc == 'BH':
        padj = p_adjust_bh(res.combined_pvalue)
    elif mtc == 'bonferroni':
        padj = np.minimum(1, res.combined_pvalue*len(res.combined_pvalue))
    res['mtc_p'] = padj
    res = res.reset_index(drop=True)
    obj.norm_data[norm]['de'][clust_alg]['combined'] = res
    if retx:
        return res


def _choose_leftright_pvalues(left, right, direction):
    """Internal helper function."""
    if direction == 'up':
        pv = right
    elif direction == 'down':
        pv = left
    else:
        pv = np.minimum(left, right)*2
    return pv


def linear_model(obj, normalization=(), clustering=(), direction='up',
                 min_cluster_size=10, verbose=False):
    """Performs differential expression analysis between clusters
    using a linear model and t-statistics

    Notes
    -----
    Finds differentially expressed (DE) genes between clusters by
    fitting a linear model (LM) to gene expression (response
    variables) and clusters (explanatory variables) and performs
    pairwise t-tests for significance. Model coefficients are
    estimated via the ordinary least squares method and p-values are
    calculated using t-statistics. One benefit of using LM for DE is
    that computations are vectorized and therefore very fast.

    The ideas behind using LM to explore DE have been extensively
    covered in the limma R package.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    normalization : `tuple`
        A tuple of normalization to use. If it has the length zero,
        then all available normalizations will be used.
    clustering : `tuple`, optional
        Specifies the clustering outcomes to work on.
    direction : `{'up', 'down', 'any'}`
        Can be 'any' for any direction 'up' for up-regulated and
        'down' for down-regulated. Normally we want to find genes
        being upregulated in cluster A compared with cluster
        B. Default: up
    min_cluster_size : `int`
        Minimum number of cells per cluster (clusters smaller than
        this are ignored).  Default: 10
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
    for _, k in enumerate(targets):
        if verbose:
            print('Running differential expression analysis on prediction on \
the %s normalization' % k)
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
                dm_full = patsy.dmatrix(
                    '~ 0 + C(cl)', pd.DataFrame({'cl': cl}))
                resid_df = dm_full.shape[0] - dm_full.shape[1]

                # gene expression should be the response
                lm = sm.regression.linear_model.OLS(endog=X_f,  # response
                                                    exog=dm_full)
                res = lm.fit()
                coef = res.params  # coefficients

                # computing standard errors
                # https://stats.stackexchange.com/questions/44838/how-are-the-standard-errors-of-coefficients-calculated-in-a-regression
                # http://web.mit.edu/~r/current/arch/i386_linux26/lib/R/library/limma/html/lm.series.html
                # residual variance for each gene
                dm_nrow = dm_full.shape[0]
                dm_ncol = dm_full.shape[1]
                sigma2 = ((X_f-dm_full.dot(coef))**2).sum(axis=0) / \
                    (dm_nrow-dm_ncol)

                q = dm_full.transpose().dot(dm_full)
                chol = np.linalg.cholesky(q)
                chol2inv = scipy.linalg.cho_solve(
                    (chol, False), np.eye(chol.shape[0]))
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
                    con[kk, ] = 1
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

                        comparisons.append('%s_vs_%s' %
                                           (clusts[kk], clusts[ii]))
                        comparisons.append('%s_vs_%s' %
                                           (clusts[ii], clusts[kk]))

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

                ll = pd.DataFrame({'comparison_A_vs_B': pd.concat(lab1,
                                                                  ignore_index=True),
                                   'gene': pd.concat(lab2, ignore_index=True),
                                   'p_val': pval,
                                   'FDR': p_adjust_bh(pval),
                                   't_stat': pd.concat(out_t_stats, ignore_index=True),
                                   'logFC': pd.concat(out_lfc, ignore_index=True),
                                   'mean.A': pd.concat(out_mge_g1, ignore_index=True),
                                   'mean.B': pd.concat(out_mge_g2, ignore_index=True)})
                obj.norm_data[k]['de'][algo] = {'long_format': ll,
                                                'mat_format': out_merged}


def _wilcox_worker(cc1, cc2, cl, X, verbose):
    """Used internally in wilcox. This function must be outside of
    wilcox() for apply_async to work."""
    if verbose:
        print('Working on cluster %s vs %s' % (cc1, cc2))
    X_ss1 = X.iloc[:, (cl == cc1).values]
    X_ss2 = X.iloc[:, (cl == cc2).values]
    z = zip(X_ss1.iterrows(), X_ss2.iterrows())
    pvs = []
    for d in z:
        try:
            pv = mannwhitneyu(d[0][1], d[1][1])
        except ValueError:
            # reaches here if all zeros in both groups, then assume
            # p-value is 1 (no difference at all)
            pv = np.nan, 1
        pvs.append(pv[1])
    pvs = pd.Series(pvs, index=X_ss1.index, name='%s_vs_%s' % (cc1, cc2))
    return pvs


def wilcox(obj, normalization=None, clust_alg=None,
           min_cluster_size=10, nworkers='auto', retx=False,
           verbose=True):
    """Performs differential expression analysis between clusters
    using the Wilcoxon rank-sum test (Mann–Whitney U test)

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    normalization : `str`
        The name of the normalization to operate on. If this is empty
        or None then the function will be applied on the last
        normalization that was applied.
    clust_alg : `str`
        Name of the clustering strategy. If empty or None, the last
        one will be used.
    min_cluster_size : `int`
        Minimum number of cells per cluster (clusters smaller than
        this are ignored).  Default: 10
    nworkers : `int` or `{'auto'}`
        If a string, then the only accepted value is 'auto', and the
        number of worker processes will be the total number of
        detected physical cores. If an integer then it specifies the
        number of worker processes. Default: 'auto'
    retx : `bool`
        Returns a data frame with results (only modifying the object
        if False).  Default: False
    verbose : `bool`
        Be verbose or not. Default: True

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    start_time = time.time()
    if type(nworkers) == str:
        if nworkers == 'auto':
            nworkers = psutil.cpu_count(logical=False)
        else:
            raise Exception('Invalid value for parameter "nworkers".')
    if verbose:
        print('%s worker processes will be used' % nworkers)
    if normalization == None or normalization == '':
        norm = list(obj.norm_data.keys())[-1]
    else:
        norm = normalization
    if verbose and normalization == None:
        print('Working on %s' % norm)
    try:
        target = obj.norm_data[norm]
    except KeyError:
        raise Exception('"%s" not found' % norm)
    if clust_alg == None or clust_alg == '':
        clust_alg = list(target['clusters'].keys())[-1]
    cl = target['clusters'][clust_alg]['membership']
    q = pd.Series(cl).value_counts()
    q = q[q >= min_cluster_size].index
    res = []
    X = target['data']
    pool = Pool(nworkers)

    def _update_results(y):
        res.append(y)

    for cc1 in q:
        for cc2 in np.arange(cc1+1, len(q)):
            args = (cc1, cc2, cl, X, verbose)
            pool.apply_async(_wilcox_worker, args=args,
                             callback=_update_results)

    pool.close()
    pool.join()
    res = pd.concat(res, axis=1)
    obj.norm_data[norm]['de'][clust_alg] = {
        'long_format': None, 'mat_format': res}
    end_time = time.time()
    if verbose:
        print('Analysis took %.2f minutes' % ((end_time-start_time)/60))
    if retx:
        return res
