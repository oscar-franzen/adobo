# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for pre-processing scRNA-seq data.
"""
import os
import sys
import glob
import ctypes
import time
import warnings
from multiprocessing import Pool

import psutil
import numpy.ctypeslib as npct
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import scale as sklearn_scale
from sklearn.linear_model import ElasticNet
from scipy.optimize import root
from scipy.special import digamma

import adobo.IO
from .clustering import knn, snn, leiden
from .hvg import seurat
from .dr import irlb
from ._log import warning

# Suppress warnings from sklearn
def _warn(*args, **kwargs):
    pass
warnings.warn = _warn

def reset_filters(obj):
    """Resets cell and gene filters

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    obj.meta_cells.status[obj.meta_cells.status != 'OK'] = 'OK'
    obj.meta_genes.status[obj.meta_genes.status != 'OK'] = 'OK'

def simple_filter(obj, what='cells', minreads=1000, maxreads=None, mingenes=None,
                  maxgenes=None, min_exp=0.001, verbose=False):
    """Removes cells with too few reads or genes with very low expression

    Notes
    -----
    Default is to remove cells.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    what : `{'cells', 'genes'}`
        Determines what should be filtered from the expression matrix. If 'cells', then
        cells are filtered. If 'genes', then genes are filtered. Default: 'cells'
    minreads : `int`, optional
        When filtering cells, defines the minimum number of reads per cell needed to
        keep the cell. Default: 1000
    maxreads : `int`, optional
        When filtering cells, defines the maximum number of reads allowed to keep the
        cell. Useful for filtering out suspected doublets. Default: None
    mingenes : `float`, `int`
        When filtering cells, defines the minimum number of genes that must be expressed
        in a cell to keep it. Default: None
    maxgenes : `float`, `int`
        When filtering cells, defines the maximum number of genes that a cell is allowed
        to express to keep it. Default: None
    min_exp : `float`, `int`    
        Used to set a threshold for how to filter out genes. If integer, defines the
        minimum number of cells that must express a gene to keep the gene. If float,
        defines the  minimum fraction of cells must express the gene to keep the gene.
        Set to None to ignore this option. Default: 0.001
    verbose : `bool`, optional
        Be verbose or not. Default: False
    
    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.preproc.simple_filter(exp, what='cells', minreads=1500)
    >>> ad.preproc.simple_filter(exp, what='genes')

    Returns
    -------
    int
        Number of removed cells or genes.
    """
    count_data = obj.count_data
    # reset
    if what == 'cells':
        obj.meta_cells.status[obj.meta_cells.status != 'OK'] = 'OK'
        cell_counts = obj.meta_cells.total_reads
        dctd_genes = obj.meta_cells.detected_genes
        if not maxreads:
            maxreads = np.max(cell_counts)
        if not minreads:
            minreads = 0
        if not mingenes:
            mingenes = np.min(dctd_genes)
        if not maxgenes:
            maxgenes = np.max(dctd_genes)
        cells_keep = np.logical_and(
                        np.logical_and(cell_counts >= minreads,
                                       cell_counts <= maxreads),
                        np.logical_and(dctd_genes >= mingenes,
                                       dctd_genes <= maxgenes)
                                   )
        obj.meta_cells.status[np.logical_not(cells_keep)] = 'EXCLUDE'
        remove = np.sum(np.logical_not(cells_keep))
    elif what == 'genes':
        obj.meta_genes.status[obj.meta_genes.status != 'OK'] = 'OK'
        if type(min_exp) == int:
            genes_exp = obj.meta_genes.expressed
            genes_remove = genes_exp < min_exp
            obj.meta_genes.status[genes_remove] = 'EXCLUDE'
        else:
            genes_exp = obj.meta_genes.expressed_perc
            genes_remove = genes_exp < min_exp
            obj.meta_genes.status[genes_remove] = 'EXCLUDE'
        remove = np.sum(genes_remove)
    if verbose:
        s = '%s cells and %s genes were removed'
        print(s % (np.sum(obj.meta_cells.status == 'EXCLUDE'),
                   np.sum(obj.meta_genes.status == 'EXCLUDE')))
    return remove

def find_mitochondrial_genes(obj, mito_pattern='^mt-', genes=None, verbose=False):
    """Find mitochondrial genes and adds percent mitochondrial expression of total
    expression to the cellular meta data

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    mito_pattern : `str`
        A regular expression matching mitochondrial gene symbols. Default: "^mt-"
    genes : `list`, optional
        Instead of using `mito_pattern`, specify a `list` of genes that are mitochondrial.
    verbose : boolean
        Be verbose or not. Default: False

    Returns
    -------
    int
        Number of mitochondrial genes detected.
    """
    count_data = obj.count_data
    if genes is None:
        mito = count_data.index.str.contains(mito_pattern, regex=True, case=False)
        obj.meta_genes['mitochondrial'] = mito
    else:
        mito = obj.count_data.index.isin(genes)
        obj.meta_genes['mitochondrial'] = mito
    no_found = np.sum(obj.meta_genes['mitochondrial'])
    if no_found > 0:
        mt = obj.meta_genes[obj.meta_genes.mitochondrial].index
        mt_counts = obj.count_data.loc[mt, :]
        mt_counts = mt_counts.sum(axis=0)
        mito_perc = mt_counts / obj.meta_cells.total_reads*100
        obj.add_meta_data(axis='cells', key='mito_perc', data=mito_perc, type_='cont')
    if verbose:
        print('%s mitochondrial genes detected' % no_found)
    obj.set_assay(sys._getframe().f_code.co_name)
    return no_found

def find_ercc(obj, ercc_pattern='^ERCC[_-]\S+$', verbose=False):
    """Flag ERCC spikes

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    ercc_pattern : `str`, optional
        A regular expression matching ercc gene symbols. Default: "ercc[_-]\S+$"
    verbose : `bool`, optional
        Be verbose or not. Default: False

    Returns
    -------
    int
        Number of detected ercc spikes.
    """
    count_data = obj.count_data
    ercc = count_data.index.str.contains(ercc_pattern)
    obj.meta_genes['ERCC'] = ercc
    obj.meta_genes['status'][ercc] = 'EXCLUDE'
    no_found = np.sum(ercc)
    obj.ercc_pattern = ercc_pattern
    if no_found > 0:
        ercc = obj.meta_genes[obj.meta_genes.ERCC].index
        ercc_counts = obj.count_data.loc[ercc, :]
        ercc_counts = ercc_counts.sum(axis=0)
        ercc_perc = ercc_counts / obj.meta_cells.total_reads*100
        obj.add_meta_data(axis='cells', key='ercc_perc', data=ercc_perc, type_='cont')
    if verbose:
        print('%s ercc spikes detected' % no_found)
    obj.set_assay(sys._getframe().f_code.co_name)
    return no_found

def find_low_quality_cells(obj, rRNA_genes, sd_thres=3, seed=42, verbose=False):
    """Statistical detection of low quality cells using Mahalanobis distances

    Notes
    ----------------
    Mahalanobis distances are computed from five quality metrics. A robust estimate of
    covariance is used in the Mahalanobis function. Cells with Mahalanobis distances of
    three standard deviations from the mean are by default considered outliers.
    The five metrics are:

        1. log-transformed number of molecules detected
        2. the number of genes detected
        3. the percentage of reads mapping to ribosomal
        4. mitochondrial genes
        5. ercc recovery (if available)

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    rRNA_genes : `list` or `str`
        Either a list of rRNA genes or a string containing the path to a file containing
        the rRNA genes (one gene per line).
    sd_thres : `float`
        Number of standard deviations to consider significant, i.e. cells are low quality
        if this. Set to higher to remove fewer cells. Default: 3
    seed : `float`
        For the random number generator. Default: 42
    verbose : `bool`
        Be verbose or not. Default: False

    Returns
    -------
    list
        A list of low quality cells that were identified, and also modifies the passed
        object.
    """

    if np.sum(obj.meta_genes.mitochondrial) == 0:
        raise Exception('No mitochondrial genes found. Run detect_mito() first.')
    if np.sum(obj.meta_genes.ERCC) == 0:
        raise Exception('No ERCC spike-ins found. Run detect_ercc_spikes() first.')
    if type(rRNA_genes) == str:
        rRNA_genes = pd.read_csv(rRNA_genes, header=None)
        obj.meta_genes['rRNA'] = obj.meta_genes.index.isin(rRNA_genes.iloc[:, 0])
    if not 'mito' in obj.meta_cells.columns:
        mt_genes = obj.meta_genes.mitochondrial[obj.meta_genes.mitochondrial]
        mito_mat = obj.count_data[obj.count_data.index.isin(mt_genes.index)]
        mito_sum = mito_mat.sum(axis=0)
        obj.meta_cells['mito'] = mito_sum
    if not 'ERCC' in obj.meta_cells.columns:
        ercc = obj.meta_genes.ERCC[obj.meta_genes.ERCC]
        ercc_mat = obj.count_data[obj.count_data.index.isin(ercc.index)]
        ercc_sum = ercc_mat.sum(axis=0)
        obj.meta_cells['ERCC'] = ercc_sum
    if not 'rRNA' in obj.meta_cells.columns:
        rrna_genes = obj.meta_genes.rRNA[obj.meta_genes.rRNA]
        rrna_mat = obj.count_data[obj.count_data.index.isin(rrna_genes.index)]
        rrna_sum = rrna_mat.sum(axis=0)
        obj.meta_cells['rRNA'] = rrna_sum
    #data = obj.count_data
    inp_total_reads = obj.meta_cells.total_reads
    inp_detected_genes = obj.meta_cells.detected_genes/inp_total_reads
    inp_rrna = obj.meta_cells.rRNA/inp_total_reads
    inp_mt = obj.meta_cells.mito/inp_total_reads
    inp_ercc = obj.meta_cells.ERCC/inp_total_reads

    qc_mat = pd.DataFrame({'reads_per_cell' : np.log(inp_total_reads),
                           'no_genes_det' : inp_detected_genes,
                           'perc_rRNA' : inp_rrna,
                           'perc_mt' : inp_mt,
                           'perc_ercc' : inp_ercc})
    robust_cov = MinCovDet(random_state=seed).fit(qc_mat)
    mahal_dists = robust_cov.mahalanobis(qc_mat)
    MD_mean = np.mean(mahal_dists)
    MD_sd = np.std(mahal_dists)
    thres_lower = MD_mean - MD_sd * sd_thres
    thres_upper = MD_mean + MD_sd * sd_thres
    res = (mahal_dists < thres_lower) | (mahal_dists > thres_upper)
    low_quality_cells = obj.count_data.columns[res].values
    obj.low_quality_cells = low_quality_cells
    obj.set_assay(sys._getframe().f_code.co_name)
    r = obj.meta_cells.index.isin(low_quality_cells)
    obj.meta_cells.status[r] = 'EXCLUDE'
    if verbose:
        print('%s low quality cell(s) identified' % len(low_quality_cells))
    return low_quality_cells

def _imputation_worker(cellids, subcount, droprate, cc, Ic, Jc, drop_thre, verbose,
                       batch):
    """A helper function for impute(...)'s multiprocessing. Don't use this function
    directly. Don't move this function below because it must be Picklable for async'ed
    usage."""
    res = []
    idx = 1
    for cellid in cellids:
        if verbose:
            v = (idx, len(cellids), batch, cc)
            print('imputing cell %s/%s (batch %s) in cluster %s' % v)
        yobs = subcount.iloc[:, cellid]
        yimpute = [0]*Ic
        nbs = set(np.arange(0, Jc))-set([cellid])
        # dropouts
        geneid_drop = droprate[:, cellid] > drop_thre
        # non-dropouts
        geneid_obs = droprate[:, cellid] <= drop_thre
        xx = subcount.iloc[geneid_obs, list(nbs)]
        yy = subcount.iloc[geneid_obs, cellid]
        ximpute = subcount.iloc[geneid_drop, list(nbs)]
        num_thre = 500
        if xx.shape[1] >= min(num_thre, xx.shape[0]):
            if num_thre >= xx.shape[0]:
                new_thre = round((2*xx.shape[0]/3))
            else:
                new_thre = num_thre
        regr = ElasticNet(random_state=0, max_iter=3000, positive=True, l1_ratio=0,
                          fit_intercept=False)
        ret = regr.fit(X=xx, y=yy.values)
        ynew = regr.predict(ximpute)
        yimpute = np.array(yimpute).astype(float)
        yimpute[geneid_drop] = ynew
        yimpute[geneid_obs] = yobs[geneid_obs]
        maxobs = [max(k) for g, k in subcount.iterrows()]
        maxobs = np.array(maxobs)
        yimpute[yimpute > maxobs] = maxobs[yimpute > maxobs]
        res.append(list(yimpute))
        idx += 1
    return [cellids, res]

def impute(obj, filtered=True, res=0.5, drop_thre=0.5, nworkers='auto', verbose=True):
    """Impute dropouts using the method described in Li (2018) Nature Communications

    Notes
    -----
    Dropouts are artifacts in scRNA-seq data. One method to alleviate the problem with
    dropouts is to perform imputation (i.e. replacing missing data points with predicted
    values).

    The present method uses a different procedure for subpopulation identification
    as compared with the original paper.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    filtered : `bool`
        If data have been filtered using :func:`adobo.preproc.simple_filter`, run
        imputation on filtered data; otherwise runs on the entire raw read count matrix.
        Default: True
    res : `float`
        Resolution parameter for the Leiden clustering, change to modify cluster
        resolution. Default: 0.5
    drop_thre : `float`
        Drop threshold. Default: 0.5
    nworkers : `int` or `{'auto'}`
        If a string, then the only accepted value is 'auto', and the number of worker
        processes will be the total number of detected physical cores. If an integer
        then it specifies the number of worker processes. Default: 'auto'
    verbose : `bool`
        Be verbose or not. Default: True

    References
    ----------
    .. [1] Li & Li (2018)
           An accurate and robust imputation method scImpute for single-cell
           RNA-seq data https://www.nature.com/articles/s41467-018-03405-7
    .. [2] https://github.com/Vivianstats/scImpute

    Returns
    -------
    Modifies the passed object.
    """
    ncores = psutil.cpu_count(logical=False)
    if type(nworkers) == str:
        if nworkers == 'auto':
            nworkers = ncores
        else:
            raise Exception('Invalid value for parameter "nworkers".')
    elif type(nworkers) == int:
        if nworkers > ncores:
            warning('"nworkers" is set to a number higher than the available number of \
physical cores on this machine (n=%s).' % ncores)
    if verbose:
        print('%s worker processes will be used' % nworkers)
    # contains normal and gamma probability density functions implemented in C (a bit
    # faster than using scipy.stats)
    time_start = time.time()
    for p in sys.path:
        pp = glob.glob('%s/pdf.*.so' % p)
        if len(pp) == 1:
            ext = ctypes.cdll.LoadLibrary(pp[0])
    ext.dgamma.argtypes = [npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS'),
                           ctypes.c_int,
                           ctypes.c_double,
                           ctypes.c_double,
                           npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')]
    ext.dnorm.argtypes = [npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS'),
                          ctypes.c_int,
                          ctypes.c_double,
                          ctypes.c_double,
                          npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')]
    # normalize
    raw = obj.count_data.copy()
    if filtered:
        # Remove low quality cells
        remove = obj.meta_cells.status[obj.meta_cells.status != 'OK']
        raw = raw.drop(remove.index, axis=1)
        # Remove uninformative genes (e.g. lowly expressed and ERCC)
        remove = obj.meta_genes.status[obj.meta_genes.status != 'OK']
        raw = raw.drop(remove.index, axis=0)
        if verbose:
            print('Running on the quality filtered data (dimensions %sx%s)' % raw.shape)
    col_sums = np.array([np.sum(i[1]) for i in raw.transpose().iterrows()])
    raw = raw*(10**6/col_sums)
    lnorm = np.log10(raw+1.01)
    lnorm_imp = lnorm
    # estimate subpopulations
    hvg = seurat(lnorm, ngenes=1000) # get hvg
    lnorm_hvg = lnorm[lnorm.index.isin(hvg)]
    d_scaled = sklearn_scale(lnorm_hvg.transpose(),  # cells as rows and genes as columns
                             axis=0,                 # over genes, i.e. features (columns)
                             with_mean=True,         # subtracting the column means
                             with_std=True)          # scale the data to unit variance
    d_scaled = pd.DataFrame(d_scaled.transpose(), index=lnorm_hvg.index)
    comp, _ = irlb(d_scaled)
    # estimating subpopulations
    nn_idx = knn(comp)
    snn_graph = snn(nn_idx)
    cl = np.array(leiden(snn_graph, res))
    nclust = len(np.unique(cl))
    if verbose:
        print('going to work on %s clusters' % nclust)

    def weight(x, params):
        inp = x
        g_out = np.zeros(len(inp))
        n_out = np.zeros(len(inp))
        # takes scale as input (rate=1/scale)
        ext.dgamma(np.array(inp), len(inp), params[1], 1/params[2], g_out)
        # SLOW (scipy.stats): dgamma.pdf(x, a=params[1], scale=1, loc=0)
        pz1 = params[0] * g_out
        ext.dnorm(np.array(inp), len(inp), params[3], params[4], n_out)
        # SLOW (scipy.stats): norm.pdf(x, params[3], params[4])
        pz2 = (1-params[0])*n_out
        pz = pz1/(pz1+pz2)
        pz[pz1 == 0] = 0
        return np.array([pz, 1-pz])

    def update_gmm_pars(x, wt):
        tp_s = np.sum(wt)
        tp_t = np.sum(wt * x)
        tp_u = np.sum(wt * np.log(x))
        tp_v = -tp_u / tp_s - np.log(tp_s / tp_t)
        if tp_v <= 0:
            alpha = 20
        else:
            alpha0 = (3 - tp_v + np.sqrt((tp_v - 3)**2 + 24 * tp_v)) / 12 / tp_v
            if alpha0 >= 20:
                alpha = 20
            else:
                alpha = root(lambda x: np.log(x) - digamma(x) - tp_v, 0.9*alpha0).x[0]
        beta = tp_s / tp_t * alpha
        return alpha, beta

    def dmix(x, pars):
        inp = x
        g_out = np.zeros(len(inp))
        n_out = np.zeros(len(inp))
        ext.dgamma(np.array(inp), len(inp), pars[1], 1/pars[2], g_out)
        #dg = dgamma(a=pars[1], scale=1/pars[2], loc=0)
        #dg.pdf(x)
        #dn = norm(pars[3], pars[4])
        # dn.pdf(x)
        ext.dnorm(np.array(inp), len(inp), pars[3], pars[4], n_out)
        return pars[0]*g_out*2+(1-pars[0])*n_out

    def para_est(x):
        params = [0, 0.5, 1, 0, 0]
        params[0] = np.sum(x == np.log10(1.01))/len(x)
        if params[0] == 0:
            params[0] = 0.01
        x_rm = x[x > np.log10(1.01)]
        params[3] = np.mean(x_rm)
        params[4] = np.std(x_rm)
        eps, iter_, loglik_old = 10, 0, 0
        while eps > 0.5:
            wt = weight(x, params)
            params[0] = np.sum(wt[0])/len(wt[0])
            params[3] = np.sum(wt[1]*x)/np.sum(wt[1])
            params[4] = np.sqrt(np.sum(wt[1]*(x-params[3])**2)/np.sum(wt[1]))
            params[1:3] = update_gmm_pars(x, wt[0])
            loglik = np.sum(np.log10(dmix(x, params)))
            eps = (loglik - loglik_old)**2
            loglik_old = loglik
            iter_ = iter_ + 1
            if iter_ > 100:
                break
        return params

    def get_par(mat, verbose):
        null_genes = np.abs(mat.sum(axis=1)-np.log10(1.01)*mat.shape[1]) < 1e-10
        null_genes = null_genes[null_genes].index
        paramlist = []
        i = 0
        for g, k in mat.iterrows():
            if verbose:
                if (i%100) == 0:
                    v = ('{:,}'.format(i), '{:,}'.format(mat.shape[0]))
                    s = 'estimating model parameters. finished with %s/%s genes' % v
                    print(s, end='\r')
            if g in null_genes:
                paramlist.append([np.nan]*5)
            else:
                paramlist.append(para_est(k.values))
            i += 1
        if verbose:
            print('\nmodel parameter estimation has finished')
        return np.array(paramlist)

    def find_va_genes(mat, parlist):
        point = np.log10(1.01)
        is_na = [not np.any(i) for i in np.isnan(np.array(parlist))]
        valid_genes = np.logical_and(mat.sum(axis=1) > point*mat.shape[1], is_na)
        return valid_genes
        #mu = parlist[:, 3]
        #sgene1 = valid_genes.index[mu<=np.log10(1+1.01)]
        #dcheck1 = dgamma.pdf(mu+1, a=parlist[:,1], scale=1/parlist[:,2], loc=0)
        #dcheck2 = norm.pdf(mu+1, parlist[:, 3], parlist[:, 4])
        #sgene3 = valid_genes.index[np.logical_and(dcheck1 >= dcheck2, mu <= 1)]
        #return valid_genes[np.logical_not(np.logical_or(sgene1,sgene3))].index

    for cc in np.arange(0, nclust):
        if verbose:
            print('estimating dropout probability for cluster %s' % cc)
        lnorm_cc = lnorm.iloc[:, cl == cc]
        # estimate model parameters
        parlist = get_par(lnorm_cc, verbose)
        if verbose:
            print('searching for valid genes for cluster %s' % cc)
        valid_genes = find_va_genes(lnorm_cc, parlist)
        if verbose:
            print('%s genes are valid' % '{:,}'.format(len(valid_genes)))
        subcount = lnorm_cc.loc[valid_genes, :]
        subcount = subcount.reindex(valid_genes[valid_genes].index)
        Ic = subcount.shape[0]
        Jc = subcount.shape[1]
        if Jc == 1:
            continue
        parlist = parlist[valid_genes]
        idx = 0
        droprate = []
        for g, k in subcount.iterrows():
            wt = weight(k, parlist[idx])[0]
            idx += 1
            droprate.append(wt)
        droprate = np.array(droprate)
        mu = parlist[:, 3]
        mucheck = subcount.apply(lambda x: x > mu, axis=0)
        droprate[np.logical_and(mucheck, droprate > drop_thre)] = 0
        # dropouts
        if verbose:
            print('running imputation for cluster %s' % cc)
        imputed = []
        pool = Pool(nworkers)

        def update_result(yimpute):
            imputed.append(yimpute)

        time_s = time.time()
        ids = np.arange(0, subcount.shape[1])
        if len(ids) < nworkers or len(ids) < 50:
            batch_size = len(ids)
        else:
            batch_size = round(len(ids)/nworkers)
        batch = 1
        while len(ids) > 0:
            ids_b = ids[0:batch_size]
            args = (ids_b, subcount, droprate, cc, Ic, Jc, drop_thre, verbose, batch)
            r = pool.apply_async(_imputation_worker,
                                 args=args,
                                 callback=update_result)
            ids = ids[batch_size:]
            batch += 1
        pool.close()
        pool.join()
        if len(imputed) == 0:
            continue
        # sorting b/c cells are not returned from subprocesses in the original order
        cellids = []
        d = []
        for item in imputed:
            cellids.append(item[0])
            d.append(item[1])
        cellids = np.concatenate(cellids)
        imputed = np.concatenate(d)
        time_e = time.time()
        if verbose:
            v = (cc, (time_e - time_s)/60)
            print('imputation for cluster %s finished in %.2f minutes' % v)
        imputed = imputed.transpose()
        imputed = pd.DataFrame(imputed)
        imputed.columns = cellids
        imputed.index = valid_genes[valid_genes].index
        imputed = imputed.sort_index(axis=1)
        lnorm_imp.loc[valid_genes, lnorm_cc.columns] = imputed.to_numpy()
    # reverse normalisation
    lnorm_imp = 10**lnorm_imp - 1.01
    lnorm_imp = lnorm_imp*(col_sums/10**6)
    lnorm_imp = round(lnorm_imp, 2)
    obj.imp_count_data = lnorm_imp
    time_end = time.time()
    if verbose:
        t = (time_end - time_start)/60
        print('imputation finished in %.2f minutes. imputed data are present in the \
"imp_count_data" attribute.' % t)
    obj.set_assay(sys._getframe().f_code.co_name)

def symbol_switch(obj, species):
    """Changes gene symbol format

    Notes
    -----
    If gene symbols are in the format ENS[0-9]+, this function changes gene identifiers
    to symbol_ENS[0-9]+.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    species : '{'human', 'mouse'}'
        Species. Default: 'human'
    
    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.preproc.symbol_switch(exp, species='human')

    Returns
    -------
    Modifies the passed object.
    """
    if not species in ('human', 'mouse'):
        raise Exception('species can be human or mouse.')
    v = (os.path.dirname(adobo.IO.__file__), species)
    gs = pd.read_csv('%s/data/%s.gencode_v32.genes.txt' % v, sep='\t', header=None)
    gs.index = gs.loc[:, 0]
    gs = gs[gs.index.isin(obj.count_data.index)]
    
    missing = obj.count_data.index[np.logical_not(obj.count_data.index.isin(gs.index))]
    gs = pd.concat([gs, pd.DataFrame({ 0 : missing, 1: ['NA']*len(missing) })])
    gs.index = gs.iloc[:, 0]
    gs = gs.reindex(obj.count_data.index)
    
    X = obj.count_data
    X.index = (gs[[1]].values+'_'+gs[[0]].values).flatten()
    obj.count_data = X
