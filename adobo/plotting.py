# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
#     Contact: Oscar Franzén <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for plotting scRNA-seq data.
"""
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import networkx as nx
import igraph as ig
import mplcursors
from scipy.cluster.hierarchy import dendrogram, linkage

import adobo
from .dr import svd, irlb
from ._constants import CLUSTER_COLORS_DEFAULT, YLW_CURRY
from ._colors import unique_colors


def _mpl_finish(filename, block=False, **args):
    """Only used internally. Used to finish off the plot."""
    plt.tight_layout()
    if filename != None:
        plt.savefig(filename, **args)
    else:
        plt.show(block=block)

def overall_scatter(obj, color_kept=YLW_CURRY, color_filtered='red',
                    no_plot=False, title=None, filename=None, **args):
    """Generates a scatter plot showing the total number of reads on
       one axis and the number of detected genes on the other axis

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    color_kept : `str`
        Color of the plot. Default: '#E69F00'
    color_filtered : `str`
        Color of the cells that have been filtered out. Default: 'red'
    no_plot : `bool`
        Don't generate a plot, only return a data frame containing
        the data points. Default: False
    title : `str`
        Title of the plot. Default: None
    filename : `str`, optional
        Write plot to file instead of showing it on the
        screen. Default: None

    Returns
    -------
    None
    """
    plt.clf()
    plt.close(fig='all')
    count_data = obj.count_data
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ff = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.get_yaxis().set_major_formatter(ff)
    ax.get_xaxis().set_major_formatter(ff)
    # summary statistics per cell
    reads = count_data.sum(axis=0)
    genes = pd.Series([np.sum(r[1] > 0)
                       for r in count_data.transpose().iterrows()])
    if no_plot:
        df = pd.DataFrame({
            'reads' : reads.values,
            'genes' : genes.values,
            'removed' : obj.meta_cells.status != 'OK'
        })
        return df
    ax.scatter(x=reads, y=genes, color=color_kept, s=2)
    reads_not_ok = reads[obj.meta_cells.status != 'OK']
    genes_not_ok = genes[(obj.meta_cells.status != 'OK').values]
    ax.scatter(x=reads_not_ok, y=genes_not_ok, color=color_filtered, s=2)
    ax.set_ylabel('detected genes')
    ax.set_xlabel('total reads')
    if title:
        ax.set_title(title)
    if np.any(reads > 100000):
        plt.xticks(rotation=90)
    _mpl_finish(filename, bbox_inches='tight', **args)


def overall(obj, what='cells', how='histogram', bin_size=100,
            cut_off=None, color=YLW_CURRY, title=None, filename=None,
            **args):
    """Generates a plot of read counts per cell or expressed genes per
    cell

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    what : `{'cells', 'genes'}`
        If 'cells' then plots the number of reads per cell. If
        'genes', then plots the number of expressed genes per
        cell. Default: 'cells'
    how : `{'histogram', 'boxplot', 'barplot', 'violin'}`
        Type of plot to generate. Default: 'histogram'
    bin_size : `int`
        If `how` is a histogram, then this is the bin size. Default: 100
    cut_off : `int`
        Set a cut off for genes or reads by drawing a red line and
        print the number of cells over and under the cut off. Only
        valid if how='histogram'.  Default: None
    color : `str`
        Color of the plot. Default: '#E69F00'
    title : `str`
        Change the default title of the plot. Default: None
    filename : `str`, optional
        Write plot to file instead of showing it on the
        screen. Default: None

    Returns
    -------
    None
    """
    if not what in ('cells', 'genes'):
        raise Exception('"what" can only be "cells" or "genes".')
    if not how in ('histogram', 'boxplot', 'barplot', 'violin'):
        raise Exception(
            '"how" can only be "histogram", "boxplot", "barplot" or "violin".')
    plt.clf()
    plt.close(fig='all')
    count_data = obj.count_data
    if what == 'cells':
        summary = count_data.sum(axis=0)
        ylab = 'raw read counts'
        xlab = 'cells'
    elif what == 'genes':
        summary = np.array([np.sum(r[1] > 0)
                            for r in count_data.transpose().iterrows()])
        ylab = 'detected genes'
        xlab = 'cells'
    colors = [color]*(len(summary))
    plt.close(fig='all')
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ff = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.get_yaxis().set_major_formatter(ff)
    ax.get_xaxis().set_major_formatter(ff)
    if how == 'barplot':
        ax.bar(np.arange(len(summary)), sorted(summary, reverse=True),
               color=colors)
        ax.set_xlabel('cells (sorted on highest to lowest)')
    elif how == 'boxplot':
        ax.boxplot(summary)
        ax.set_xlabel('cells')
        ax.set_xticklabels([])
        ax.set_xticks([])
    elif how == 'histogram':
        ax.hist(summary, bins=bin_size, color=color)
        ax.set_xlabel('cells')
        ylab = 'frequency'
        xlab = '%s bin' % what
        if cut_off:
            ax.axvline(linewidth=1, color='r', x=cut_off)
            above = np.sum(summary > cut_off)
            below = np.sum(summary <= cut_off)
            ax.text(x=cut_off, y=1, s='%s above\n%s below' % (above, below))
    elif how == 'violin':
        parts = ax.violinplot(summary, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
        xlab = ''
        ax.set_xticklabels([])
        ax.set_xticks([])
    else:
        raise Exception('The `how` parameter has an invalid value.')
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    if not title and what == 'reads':
        title = 'total number of reads per cell'
    elif not title and what == 'genes':
        title = 'total number of expressed genes per cell'
    ax.set_title(title)
    if np.any(summary > 100000):
        plt.xticks(rotation=90)
    _mpl_finish(filename, bbox_inches='tight', **args)


def pca_contributors(obj, normalization=None, how='heatmap',
                     clust_alg=None, cluster=None, all_genes=False,
                     dim=[0, 1, 2], top=20, color=YLW_CURRY,
                     fontsize=6, figsize=(10, 5), filename=None,
                     verbose=False, **args):
    """Examine the top contributing genes to each PCA
    component. Optionally, one can examine the PCA components of a
    cell cluster instead.

    Note
    ----
    The function takes half the genes with top negative scores and the
    other half from genes with positive scores. Additional parameters
    are passed into :py:func:`matplotlib.pyplot.savefig`.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    normalization : `str`
        The name of the normalization to operate on. If empty or None,
        the last one generated is be used. Default: None
    how : `{'heatmap', 'barplot'}`
        How to visualize, can be barplot or heatmap. If 'barplot',
        then shows the PCA scores. If 'heatmap', then visualizes the
        expression of genes with top PCA scores.  Default: 'barplot'
    clust_alg : `str`
        Name of the clustering strategy. If empty or None, the last
        one generated is be used. Default: None
    cluster : `int`
        Name of the cluster.
    all_genes : `bool`
        If `cluster` is set, then indicates if PCA should be computed
        on all genes or only on the highly variable genes. Default:
        False
    dim : `list` or `int`
        If list, then it specifies indices of components to plot. If
        integer, then it specifies the first components to plot. First
        component has index zero.  Default: [0, 1, 2]
    top : `int`
        Specifies the number of top scoring genes to include
        (i.e. will use this many positive/negative scoring
        genes). Default: 20
    color : `str`
        Color of the bars. Default: "#fcc603"
    fontsize : `int`
        Specifies font size. Default: 6
    figsize : `tuple`
        Figure size in inches. Default: (10, 10)
    filename : `str`, optional
        Write to a file instead of showing the plot on screen. File
        type is determined by the filename extension.
    verbose : `bool`
        Be verbose or not. Default: False

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp, method='standard')
    >>> ad.hvg.find_hvg(exp)
    >>> ad.dr.pca(exp)
    >>> ad.plotting.pca_contributors(exp, dim=4)
    >>> # decomposition of a specific cluster
    >>> ad.clustering.generate(exp, clust_alg='leiden')
    >>> ad.plotting.pca_contributors(exp, dim=4, cluster=0)

    Returns
    -------
    Nothing
    """
    if not how in ('barplot', 'heatmap'):
        raise ValueError('"how" can only be "barplot" or "heatmap"')
    if normalization == None or normalization == '':
        norm = list(obj.norm_data.keys())[-1]
    else:
        norm = normalization
    try:
        target = obj.norm_data[norm]
    except KeyError:
        raise Exception('"%s" not found' % norm)
    if clust_alg == None or clust_alg == '':
        try:
            clust_alg = list(target['clusters'].keys())[-1]
        except IndexError:
            pass
    plt.clf()
    plt.close(fig='all')
    if not isinstance(dim, list):
        dim = [dim]
    f, ax = plt.subplots(nrows=1, ncols=len(dim), figsize=figsize)
    if type(ax) != np.ndarray:
        ax = [ax]
    f.subplots_adjust(wspace=1)
    try:
        contr = target['dr']['pca']['contr']
        comp = target['dr']['pca']['comp']
    except KeyError:
        raise Exception('PCA decomposition not found.')
    if cluster != None:
        X = target['data']
        try:
            cl = target['clusters'][clust_alg]['membership']
        except KeyError:
            raise Exception('"%s" not found, run adobo.clustering.\
generate(...)' % clust_alg)
        X_ss = X.loc[:, (cl == cluster).to_numpy()]
        if not all_genes:
            hvg = target['hvg']['genes']
            X_ss = X_ss[X_ss.index.isin(hvg)]
        comp, contr = irlb(X_ss)
    contr = contr[dim]
    idx = 0
    if how == 'barplot':
        for i, d in contr.iteritems():
            d = d.sort_values(ascending=False)
            d_top = d.head(top)
            d_bottom = d.tail(top)
            zx = pd.concat([d_top, d_bottom])
            y_pos = np.arange(len(zx))
            ax[idx].barh(y_pos, zx.values, color=YLW_CURRY)
            ax[idx].set_yticks(y_pos)
            ax[idx].set_yticklabels(zx.index.values, fontsize=fontsize)
            ax[idx].set_xlabel('abs(PCA score)', fontsize=fontsize)
            ax[idx].tick_params(labelsize=fontsize)
            ax[idx].set_title('comp. %s' % (i+1), fontsize=fontsize)
            ax[idx].invert_yaxis()  # labels read top-to-bottom
            idx += 1
    elif how == 'heatmap':
        X = target['data']
        for i, d in contr.iteritems():
            if verbose:
                print(i)
            d = d.sort_values(ascending=False)
            d_top = d.head(top)
            d_bottom = d.tail(top)
            zx = pd.concat([d_top, d_bottom])
            X_ss = X[X.index.isin(zx.index)]
            X_ss = X_ss.sparse.to_dense().reindex(zx.index)
            X_ss = X_ss.reindex(comp.iloc[:, i].sort_values().index, axis=1)
            hm = sns.heatmap(X_ss, ax=ax[idx], yticklabels=X_ss.index.values)
            hm.set_title('comp. %s' % (i+1), fontsize=fontsize)
            hm.set_xlabel('', fontsize=fontsize)
            hm.set_ylabel('', fontsize=fontsize)
            hm.tick_params(labelsize=fontsize)
            hm.set(xticklabels=[], xticks=[])
            # <BUGFIX>
            # fix for mpl bug that cuts off top/bottom of seaborn viz
            b, t = ax[idx].get_ylim()  # discover the values for bottom and top
            b += 0.5  # Add 0.5 to the bottom
            t -= 0.5  # Subtract 0.5 from the top
            ax[idx].set_ylim(b, t)  # update the ylim(bottom, top) values
            # </BUGFIX>
            idx += 1
    _mpl_finish(filename, **args)


def cell_viz(obj, reduction=None, normalization=(), clustering=(),
             metadata=(), genes=(), highlight=None,
             highlight_color=('black', 'red'), selection_mode=False,
             edges=False, cell_types=False, trajectory=None,
             filename=None, marker_size=0.8, font_size=8,
             colors='adobo', title=None, legend=True,
             legend_marker_scale=10, legend_position=(1, 1),
             min_cluster_size=10, figsize=(10, 10), margins=None,
             dark=False, aspect_ratio='equal', verbose=False, **args):
    """Generates a 2d scatter plot from an embedding

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    reduction : `{'tsne', 'umap', 'pca', 'force_graph'}`
        The dimensional reduction to use. Default is to use the last
        one generated.
    normalization : `tuple`
        A tuple of normalization to use. If it has the length zero,
        then the last generated will be used.
    clustering : `tuple`
        Specifies the clustering outcomes to plot. If None, then the
        last generated clustering is plotted.
    metadata : `tuple`, optional
        Specifies the metadata variables to plot.
    genes : `tuple`, optional
        Specifies genes to plot. Can also be a regular expression
        matching a single gene name.
    highlight : `int` or `str`
        Highlight a cluster or a single cell. Integer if cluster and
        string if a cell.
    highlight_color : `tuple`
        The colors to use when highlighting a cluster. Should be a
        tuple of length two.  First item is the color of all other
        cluster than the selected, the second item is the color of the
        highlighted cluster.
    selection_mode : `bool`
        Enables interactive selection of cells. Prints the IDs of the
        cells inside the rectangle. Default: False
    edges : `bool`
        Draw edges (only applicable if
        reduction='force_graph'). Default: False
    cell_types : `bool`
        Print cell type predictions, applicable if
        :py:func:`adobo.bio.cell_type_predict` has been run. Default:
        False
    trajectory : `str`, optional
       The trajectory to plot. For example 'slingshot'. Default: None
    filename : `str`, optional
        Name of an output file instead of showing on screen.
    marker_size : `float`
        The size of the markers. Default: 0.8
    font_size : `float`
        Font size. Default: 8
    colors : `{'default', 'random'}` or `list`
        Can be: (i) "adobo" or "random"; or (ii) a `list` of colors
        with the same length as the number of factors. If colors is
        set to "adobo", then colors are retrieved from
        :py:attr:`adobo._constants.CLUSTER_COLORS_DEFAULT` (but if the
        number of clusters exceed 50, then random colors will be
        used). Default: adobo
    title : `str`
        An optional title of the plot.
    legend : `bool`
        Add legend or not. Default: True
    legend_marker_scale : `int`
        Scale the markers in the legend. Default: 10
    legend_position : `tuple`
        A tuple of length two describing the position of the
        legend. Default: (1,1)
    min_cluster_size : `int`
        Can be used to prevent clusters below a certain number of
        cells to be plotted. Default: 10
    figsize : `tuple`
        Figure size in inches. Default: (10, 10)
    margins : `dict`
        Can be used to adjust margins. Should be a dict with one or
        more of the keys: 'left', 'bottom', 'right', 'top', 'wspace',
        'hspace'. Set verbose=True to figure out the present
        values. Default: None
    dark : `bool`
        Make the background color black. Default: False
    aspect_ratio : `{'equal', 'auto'}`
        Set the aspect of the axis scaling, i.e. the ratio of y-unit
        to x-unit. Default: 'equal'
    verbose : `bool`
        Be verbose or not. Default: True

    Returns
    -------
    None
    """
    avail_reductions = ('tsne', 'umap', 'pca', 'force_graph', None)
    D = obj.norm_data
    if not reduction in avail_reductions:
        raise Exception('`reduction` must be one of %s.' %
                        ', '.join(avail_reductions))
    if marker_size < 0:
        raise Exception('`marker_size` cannot be negative.')
    if type(clustering) == str:
        clustering = (clustering,)
    if type(metadata) == str:
        metadata = (metadata,)
    if type(genes) == str:
        genes = (genes,)
    if type(normalization) == str:
        normalization = (normalization,)
    if len(clustering) == 0 and len(metadata) == 0:
        try:
            clustering = tuple(
                {'q': list(D[item]['clusters'].keys()) for item in D}['q'])
        except KeyError:
            raise Exception(
                'No clusterings found. Run `adobo.clustering.generate` first.')
    if len(clustering) == 0 and len(metadata) == 0:
        raise Exception(
            'No clusterings found. Run `adobo.clustering.generate` first.')
    # setup colors
    if colors == 'adobo':
        colors = CLUSTER_COLORS_DEFAULT
    elif colors == 'random':
        colors = unique_colors(len(groups))
        if verbose:
            print('Using random colors: %s' % colors)
    else:
        colors = colors
    n_plots = len(clustering) + len(metadata) + len(genes)  # per row
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    if dark:
        plt.style.use('dark_background')
    targets = {}
    if len(normalization) == 0:
        targets = tuple(D.keys())
    else:
        targets = normalization
    # setup plotting grid
    plt.clf()
    plt.close(fig='all')
    fig, aa = plt.subplots(nrows=len(targets), ncols=n_plots, figsize=figsize,
                           constrained_layout=True)
    if isinstance(aa, np.ndarray):
        if aa.ndim > 1:
            aa = aa.flatten()
    else:
        aa = np.array([aa])
    if title:
        fig.suptitle(title)
    pl_idx = 0
    for _, norm_name in enumerate(targets):
        item = D[norm_name]
        if verbose:
            print(pl_idx, norm_name)
        # the embedding
        if len(item['dr'].keys()) == 0:
            q = 'Reduction "%s" was not found. Run `adobo.dr.tsne(...)` or \
`adobo.dr.umap(...)` first.' % reduction
            raise Exception(q)
        if not reduction:
            reduction = list(item['dr'].keys())[-1]
        if reduction == 'pca':
            red_key = 'comp'
        elif reduction == 'force_graph':
            red_key = 'coords'
        else:
            red_key = 'embedding'
        if reduction == 'force_graph' and edges and not 'graph' in item:
            raise Exception('Graph has not been generated. Run \
`adobo.clustering.generate(...)` first.')
        E = item['dr'][reduction][red_key]
        markerscale = legend_marker_scale
        # plot clusterings
        for cl_algo in clustering:
            if not cl_algo in item['clusters']:
                raise Exception('Clustering "%s" not found.' % cl_algo)
            cl = item['clusters'][cl_algo]['membership']
            groups = np.unique(cl)
            z = pd.Series(dict(Counter(cl)))
            if min_cluster_size > 0:
                remove = z[z < min_cluster_size].index.values
                groups = groups[np.logical_not(pd.Series(groups).isin(remove))]
            z = z[z.index.isin(groups)].sort_index()

            for i, k in enumerate(groups):
                idx = np.array(cl) == k
                e = E[idx]
                if type(highlight) == int and highlight == k:
                    col = highlight_color[1]
                elif type(highlight) == int and highlight != k:
                    col = highlight_color[0]
                elif type(highlight) == str and highlight in e.index:
                    col = np.where(e.index == highlight, highlight_color[1],
                                   highlight_color[0])
                elif type(highlight) == str and not highlight in e.index:
                    col = highlight_color[0]
                else:
                    col = colors[i]
                aa[pl_idx].scatter(e.iloc[:, 0], e.iloc[:, 1], s=marker_size,
                                   color=col, label=k)  # don't remove label, it is
                                                    # needed for sorting items
                                                    # in the legend
            aa[pl_idx].set_title('%s %s %s' % (norm_name, cl_algo, reduction),
                                 size=font_size)
            aa[pl_idx].set_aspect(aspect_ratio)
            aa[pl_idx].set_gid(reduction+'_'+norm_name+'_' + cl_algo)

            def _hh(sel):
                foo = sel[0].axes.get_gid().split('_')
                _red, _norm_name, _cl_algo = foo
                cl_i = obj.norm_data[_norm_name]['clusters'][_cl_algo]['membership']
                if cell_types:
                    key = 'cell_type_prediction'
                    ct_i = obj.norm_data[_norm_name]['clusters'][_cl_algo][key]
                E_i = obj.norm_data[_norm_name]['dr'][_red]['embedding']
                v = np.logical_and(E_i.iloc[:, 0] == sel.target[0],
                                   E_i.iloc[:, 1] == sel.target[1])
                c_idx = np.arange(0, E_i.shape[0])[v]
                cl_target = cl_i[c_idx[0]]
                if cell_types:
                    ct_target = ct_i[ct_i.index ==
                                     cl_target].loc[:, 'cell type'].values[0]
                    lab = 'cluster: %s\ncell type: %s' % (cl_target, ct_target)
                else:
                    lab = 'cluster %s' % cl_target
                sel.annotation.set_text(lab)

            def _select_callback(eclick, erelease):
                foo = eclick.inaxes.get_gid().split('_')
                _red, _norm_name, _cl_algo = foo
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
                E_i = obj.norm_data[_norm_name]['dr'][_red]['embedding']
                cells_sel = np.logical_and(
                    np.logical_and(E_i.iloc[:, 0] > x1, E_i.iloc[:, 0] < x2),
                    np.logical_and(E_i.iloc[:, 1] > y1, E_i.iloc[:, 1] < y2)
                )
                print(cells_sel[cells_sel].index)
            if not selection_mode:
                mplcursors.cursor(aa[pl_idx]).connect('add', _hh)
            else:
                rs = RectangleSelector(aa[pl_idx], _select_callback,
                                       drawtype='box', useblit=False,
                                       button=[1], minspanx=5,
                                       minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
            if pl_idx == 0:
                aa[pl_idx].set_ylabel('%s 1' % reduction)
                aa[pl_idx].set_xlabel('%s 2' % reduction)
            if legend:
                lab = (z.index.astype(str)+' (n='+z.astype(str)+')').values
                aa[pl_idx].legend(lab, loc='upper left', markerscale=markerscale,
                                  bbox_to_anchor=legend_position,
                                  prop={'size': font_size})
                if cell_types:
                    try:
                        d = obj.norm_data
                        ctp = d[norm_name]['clusters'][cl_algo]['cell_type_prediction']
                        cur_hands, cur_labs = aa[pl_idx].get_legend_handles_labels(
                        )
                        ct = ctp[['cell type']].values.flatten()
                        z = zip(groups,
                                ctp[['cell type']].values.flatten(),
                                ctp[['p-value']].values.flatten()
                                )
                        lab = [str(q[0])+', '+q[1]+', p=' +
                               str('{:.2E}'.format(q[2])) for q in z]
                        z = zip(cur_hands, cur_labs, lab, ct)
                        sl = [tup for tup in sorted(z, key=lambda x: x[3])]
                        aa[pl_idx].legend(np.array(sl)[:, 0],
                                          np.array(sl)[:, 2],
                                          loc='upper left',
                                          markerscale=markerscale,
                                          bbox_to_anchor=legend_position,
                                          prop={'size': font_size})
                    except KeyError:
                        print('cell_types is set to True, but \
adobo.bio.cell_type_predict has not been called yet.')
            if trajectory == 'slingshot':
                # cluster weights matrix
                l = np.array([(cl == clID).astype(int) for clID in groups])
                l = l.transpose()
                centers = []
                for clID in groups:
                    w = l[:, clID]
                    centers.append(np.average(E, axis=0, weights=w))
                centers = np.array(centers)
                aa[pl_idx].plot(centers[:, 0], centers[:, 1],
                                'bo', color='blue')
                adj = obj.norm_data['standard']['slingshot'][cl_algo]['adjacency']
                for i in np.arange(0, max(groups)+1):
                    for j in np.arange(i, max(groups))+1:
                        if adj.iloc[i, j] or adj.iloc[j, i]:
                            xy = centers[(i, j), :]
                            aa[pl_idx].plot(xy[:, 0], xy[:, 1], color='black')
            if edges and reduction == 'force_graph':
                d = {}
                for k, i in E.iterrows():
                    d[k] = i.values
                snn_graph = item['graph']
                nn = set(snn_graph[snn_graph.columns[0]])
                g = ig.Graph()
                g.add_vertices(len(nn))
                g.vs['name'] = list(range(1, len(nn)+1))
                ll = []
                for i in snn_graph.itertuples(index=False):
                    ll.append(tuple(i))
                g.add_edges(ll)
                A = g.get_edgelist()
                GG = nx.Graph(A)
                edge_collection = nx.draw_networkx_edges(GG, d,
                                                         ax=aa[pl_idx],
                                                         width=0.3,
                                                         edge_color='grey',
                                                         alpha=0.5)
                edge_collection.set_zorder(-2)
            pl_idx += 1
        # plot meta data variables
        for meta_var in metadata:
            if not meta_var in obj.meta_cells.columns:
                raise ValueError(
                    'Meta data variable "%s" not found.' % meta_var)
            m_d = obj.meta_cells.loc[obj.meta_cells.status == 'OK', meta_var]
            if m_d.dtype.name == 'category':
                groups = np.unique(m_d)
                for i, k in enumerate(groups):
                    idx = np.array(m_d) == k
                    e = E[idx]
                    col = colors[i]
                    aa[pl_idx].scatter(e.iloc[:, 0], e.iloc[:, 1],
                                       s=marker_size, color=col)
                if legend:
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    aa[pl_idx].legend(list(groups), loc='upper left',
                                      markerscale=markerscale, bbox_to_anchor=(
                                          1, 1),
                                      prop={'size': font_size})
            else:
                # If data are continuous
                cmap = sns.cubehelix_palette(as_cmap=True)
                po = aa[pl_idx].scatter(E.iloc[:, 0], E.iloc[:, 1],
                                        s=marker_size, c=m_d.values,
                                        cmap=cmap)
                divider = make_axes_locatable(aa[pl_idx])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(po, cax=cax)
            aa[pl_idx].set_title(meta_var, size=font_size)
            aa[pl_idx].set_aspect(aspect_ratio)
            pl_idx += 1
        # plot genes
        for gene in genes:
            if not np.any(item['data'].index.str.fullmatch(gene)):
                m = '"%s" was not found in the gene expression matrix' % gene
                raise Exception(m)
            if np.sum(item['data'].index.str.fullmatch(gene)) > 1:
                raise Exception(
                    'Multiple genes found with the name "%s"' % gene)
            #ge = item['data'].loc[gene, :]
            ge = item['data'][item['data'].index.str.fullmatch(gene)]
            cmap = sns.cubehelix_palette(as_cmap=True)
            po = aa[pl_idx].scatter(E.iloc[:, 0], E.iloc[:, 1], s=marker_size,
                                    c=ge.values[0], cmap=cmap)
            divider = make_axes_locatable(aa[pl_idx])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(po, cax=cax)
            aa[pl_idx].set_title(ge.index[0], size=font_size)
            aa[pl_idx].set_aspect(aspect_ratio)
            pl_idx += 1
        # turn off unused axes
        # if (len(clustering) + len(metadata) + len(genes)) == 1:
        #    aa[row][1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if verbose:
        print('fig.subplotpars.top: %s' % fig.subplotpars.top)
        print('fig.subplotpars.bottom: %s' % fig.subplotpars.bottom)
        print('fig.subplotpars.left: %s' % fig.subplotpars.left)
        print('fig.subplotpars.right: %s' % fig.subplotpars.right)
        print('fig.subplotpars.hspace: %s' % fig.subplotpars.hspace)
        print('fig.subplotpars.wspace: %s' % fig.subplotpars.wspace)
    if margins:
        fig.subplots_adjust(**margins)
    if dark:
        # go back to default style
        plt.style.use('classic')
        plt.rcParams.update(plt.rcParamsDefault)
    # selection mode doesn't work if block=False
    _mpl_finish(filename, block=selection_mode, **args)


def pca_elbow(obj, normalization=None, comp_max=100, all_genes=False,
              filename=None, font_size=8, figsize=(6, 4),
              color=YLW_CURRY, title='PCA elbow plot', **args):
    """Generates a PCA elbow plot

    Notes
    -----
    Can be useful for determining the number of components to
    include. Here, PCA is computed using singular value decomposition.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    normalization : `str`
        The name of the normalization to operate on. If empty or None,
        the last one generated is be used. Default: None
    comp_max : `int`
        Maximum number of components to include. Default: 100
    all_genes : `bool`
        Run on all genes, i.e. not only highly variable
        genes. Default: False
    filename : `str`, optional
        Name of an output file instead of showing on screen.
    font_size : `float`
        Font size. Default: 8
    figsize : `tuple`
        Figure size in inches. Default: (10, 10)
    color : `str`
        Color of the line. Default: #fcc603
    title : `str`
        A plot title.

    Returns
    -------
    Nothing.
    """
    if normalization == None or normalization == '':
        norm = list(obj.norm_data.keys())[-1]
    else:
        norm = normalization
    try:
        target = obj.norm_data[norm]
    except KeyError:
        raise Exception('"%s" not found' % norm)
    plt.clf()
    plt.close(fig='all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    X = target['data']
    if not all_genes:
        try:
            hvg = target['hvg']['genes']
        except KeyError:
            raise Exception('Run adobo.dr.find_hvg() first.')
        X = X[X.index.isin(hvg)]
    else:
        if verbose:
            print('Using all genes')
    sdev = svd(X, only_sdev=True)
    var = sdev**2
    pvar = var/sum(var)
    cs = np.cumsum(pvar)[0:comp_max]
    ax.plot(cs, color=color)
    ax.set_ylabel('cumulative variance (percent of total)')
    ax.set_xlabel('components')
    ax.set_title(title)
    _mpl_finish(filename, **args)


def genes_violin(obj, normalization='', clust_alg=None, cluster=None,
                 gene=None, rank_func=np.median, top=10, violin=True,
                 scale='width', fontsize=10, figsize=(10, 5),
                 linewidth=0.5, filename=None, **args):
    """Plot individual genes using violin plot (or box plot). Can be
    used to plot the top genes in the total dataset or top genes in
    individual clusters. Specific genes can also be selected using the
    parameter `genes`.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    normalization : `str`
        The name of the normalization to operate on. If this is empty
        or None then the function will be applied on the last
        normalization that was applied.
    clust_alg : `str`
        Name of the clustering strategy. If empty or None, the last
        one will be used.
    cluster : `list` or `int`
        List of cluster identifiers to plot. If a list, then expecting
        a list of cluster indices. An integer specifies only one
        cluster index. If None, then shows the expression across all
        clusters. Default: None
    gene : `str`
        Compare a single gene across all clusters (can also be a
        regular expression, but it must match a single gene). If this
        is None, then the top is plotted based on the ranking function
        specified below. Default: None
    rank_func : `np.median`
        Ranking function. numpy's median is the default.
    top : `int`
        Specifies the number of top scoring genes to include. Default: 10
    violin : `bool`
        Draws a violin plot (otherwise a box plot). Default: True
    scale : `{'width', 'area'}`
        If `area`, each violin will have the same area. If ``width``,
        each violin will have the same width. Default: 'width'
    fontsize : `int`
        Specifies font size. Default: 6
    figsize : `tuple`
        Figure size in inches. Default: (10, 10)
    linewidth : `float`
        Border width. Default: 0.5
    filename : `str`, optional
        Write to a file instead of showing the plot on screen.
    **args
        Passed on into seaborn's violinplot and boxplot functions

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp, method='standard')
    >>> ad.hvg.find_hvg(exp)
    >>> ad.dr.pca(exp)
    >>> ad.clustering.generate(exp, clust_alg='leiden')
    >>>
    >>> # top 10 genes in cluster 0
    >>> ad.plotting.genes_violin(exp, top=10, cluster=0)
    >>>
    >>> # top 10 genes across all clusters
    >>> ad.plotting.genes_violin(exp, top=10)
    >>>
    >>> # plotting one gene across all clusters
    >>> ad.plotting.genes_violin(exp, gene='ENSG00000163220')
    >>>
    >>> # same, but using a box plot
    >>> ad.plotting.genes_violin(exp, gene='ENSG00000163220', violin=False)

    Returns
    -------
    Nothing
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
        try:
            clust_alg = list(target['clusters'].keys())[-1]
        except IndexError:
            pass
    # setup plotting grid
    plt.clf()
    plt.close(fig='all')
    rows = 1
    if isinstance(cluster, int):
        cluster = [cluster]
    if cluster != None and gene == None:
        rows = len(cluster)
    fig, aa = plt.subplots(nrows=rows,
                           ncols=1,
                           figsize=figsize)
    if rows == 1:
        aa = [aa]
    X = target['data']
    if cluster != None or gene:
        try:
            cl = target['clusters'][clust_alg]['membership']
        except KeyError:
            raise Exception('Clustering %s not found' % clust_alg)
        cl = cl.values
    else:
        cl = np.array([0]*X.shape[1])
    ret = X.groupby(cl, axis=1).aggregate(rank_func)
    if cluster != None:
        if np.any([i > ret.shape[1] for i in cluster]):
            raise Exception('Wrong cell cluster index specified.')
        ret = ret[cluster]
    idx = 0
    if not gene:
        for i, d in ret.iteritems():
            d = d.sort_values(ascending=False)
            d = d.head(top)
            X_ss = X[X.index.isin(d.index)]
            X_ss = X_ss.loc[:, cl == i]
            if violin:
                p = sns.violinplot(ax=aa[idx], data=X_ss.transpose(),
                                   linewidth=linewidth, order=d.index,
                                   scale=scale, **args)
            else:
                p = sns.boxplot(ax=aa[idx], data=X_ss.transpose(),
                                linewidth=linewidth, order=d.index,
                                **args)
            p.set_xticklabels(labels=d.index, rotation=90, fontsize=fontsize)
            if cluster != None and len(cluster) > 0:
                p.set_title('cluster %s' % i)
            else:
                p.set_title('across all clusters')
            idx += 1
    else:
        if np.sum(X.index.str.match(gene)) == 1:
            X_ss = X[X.index.str.match(gene)].values[0]
            g = X.index[X.index.str.match(gene)][0]
            print(g)
        else:
            raise Exception('The gene "%s" was not found.' % gene)
        if violin:
            p = sns.violinplot(ax=aa[0], y=X_ss, x=cl,
                               linewidth=linewidth, fontsize=fontsize,
                               scale=scale, **args)
        else:
            p = sns.boxplot(ax=aa[0], y=X_ss, x=cl,
                            linewidth=linewidth, **args)
        p.set_title(g)
    p.set_ylabel('expression')
    p.set_xlabel('cluster')
    _mpl_finish(filename, **args)


def tree(obj, normalization='', clust_alg=None, method='complete',
         cell_types=True, min_cluster_size=10, fontsize=8,
         figsize=(10, 5), filename=None, title=None, **args):
    """Generates a dendrogram of cluster relationships

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    normalization : `str`
        The name of the normalization to operate on. If this is empty
        or None then the function will be applied on the last
        normalization that was applied.
    clust_alg : `str`
        Name of the clustering strategy. If empty or None, the last
        one will be used.
    method : `'{'complete', 'single', 'average', 'weighted', 'centroid', 'median', 'ward'}'`
        The linkage algorithm to use. Default: 'complete'
    cell_types : `bool`
        Add putative cell type annotations (if available). Default: True
    min_cluster_size : `int`
        Can be used to prevent clusters below a certain number of
        cells to be plotted. Default: 10
    fontsize : `int`
        Specifies font size. Default: 6
    figsize : `tuple`
        Figure size in inches. Default: (10, 10)
    filename : `str`
        Write to a file instead of showing the plot on
        screen. Default: None
    title : `str`
        Plot title.

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.preproc.symbol_switch(exp, species='human')
    >>> ad.normalize.norm(exp, method='standard')
    >>> ad.hvg.find_hvg(exp)
    >>> ad.dr.pca(exp)
    >>> ad.clustering.generate(exp, clust_alg='leiden')
    >>> ad.bio.cell_type_predict(exp, verbose=True)
    >>> ad.plotting.tree(exp)

    Returns
    -------
    Nothing
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
    # setup plotting grid
    plt.clf()
    plt.close(fig='all')
    cl = target['clusters'][clust_alg]['membership']
    z = pd.Series(dict(Counter(cl)))
    X = target['data']
    if min_cluster_size > 0:
        remove = z[z < min_cluster_size].index.values
        X = X.loc[:, np.logical_not(cl.isin(remove))]
        cl = cl[np.logical_not(cl.isin(remove))]
    ret = X.groupby(cl.values, axis=1).aggregate(np.mean)
    if cell_types:
        try:
            ctp = obj.norm_data[norm]['clusters'][clust_alg]['cell_type_prediction']
            i = np.intersect1d(ret.columns, ctp.index)
            ctp = ctp[ctp.index.isin(i)]
            ret = ret.loc[:, ret.columns.isin(i)]
            z = zip(list(map(str, ctp.index.values)), ctp.iloc[:, 1])
            ret.columns = [q[0]+'_'+q[1] for q in z]
        except KeyError:
            pass

    ret = ret.transpose()
    link = linkage(ret, method=method)
    fig, aa = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    r = dendrogram(link, orientation='top',
                   distance_sort='descending',
                   show_leaf_counts=True,
                   labels=ret.index,
                   leaf_rotation=90,
                   ax=aa,
                   leaf_font_size=fontsize,
                   link_color_func=None,
                   leaf_label_func=None,
                   color_threshold=0.01,
                   truncate_mode=None,
                   above_threshold_color='#000000')
    if not title:
        title = 'dendrogram of clusters; (%s, %s, %s)' % (
            norm, clust_alg, method)
    aa.set_title(title)
    _mpl_finish(filename, **args)


def exp_genes(obj, normalization=None, clust_alg=None, cluster=None,
              min_cluster_size=10, violin=True, scale='width',
              fontsize=10, figsize=(10, 5), linewidth=0.5,
              filename=None, title=None, **args):
    """Compare number of expressed genes across clusters

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    normalization : `str`
        The name of the normalization to operate on. If this is empty
        or None then the function will be applied on the last
        normalization that was applied.
    clust_alg : `str`
        Name of the clustering strategy. If empty or None, the last
        one will be used.
    cluster : `list` or `int`
        List of cluster identifiers to plot. If a list, then expecting
        a list of cluster indices. An integer specifies only one
        cluster index. If None, then shows the expression across all
        clusters. Default: None
    min_cluster_size : `int`
        Can be used to prevent clusters below a certain number of
        cells to be plotted. Default: 10
    violin : `bool`
        Draws a violin plot (otherwise a box plot). Default: True
    scale : `{'width', 'area'}`
        If `area`, each violin will have the same area. If ``width``,
        each violin will have the same width. Default: 'width'
    fontsize : `int`
        Specifies font size. Default: 6
    figsize : `tuple`
        Figure size in inches. Default: (10, 10)
    linewidth : `float`
        Border width. Default: 0.5
    filename : `str`, optional
        Write to a file instead of showing the plot on screen.
    title : `str`
        Title of the plot. Default: None
    **args
        Passed on into seaborn's violinplot and boxplot functions

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp, method='standard')
    >>> ad.hvg.find_hvg(exp)
    >>> ad.dr.pca(exp)
    >>> ad.clustering.generate(exp, clust_alg='leiden')
    >>> ad.plotting.exp_genes(obj)

    Returns
    -------
    Nothing
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
    # setup plotting grid
    plt.clf()
    plt.close(fig='all')
    cl = target['clusters'][clust_alg]['membership']
    z = pd.Series(dict(Counter(cl)))
    X = target['data']
    if min_cluster_size > 0:
        remove = z[z < min_cluster_size].index.values
        X = X.loc[:, np.logical_not(cl.isin(remove))]
        cl = cl[np.logical_not(cl.isin(remove))]
    cl_names = []
    cl_exp_genes = []
    for cl_id, X_ss in X.groupby(cl.values, axis=1):
        g_exp = (X_ss > 0).sum(axis=0)
        cl_names = cl_names + [cl_id]*len(g_exp)
        cl_exp_genes = cl_exp_genes + list(g_exp.values)
    fig, aa = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if violin:
        p = sns.violinplot(ax=aa, x=cl_names, y=cl_exp_genes,
                           linewidth=linewidth, scale=scale, **args)
    else:
        p = sns.boxplot(ax=aa, x=cl_names, y=cl_exp_genes,
                        linewidth=linewidth, **args)
    p.set_ylabel('number of expressed genes')
    p.set_xlabel('cluster')
    if title:
        p.set_title(title)
    _mpl_finish(filename, **args)


def jackstraw_barplot(obj, normalization=None, fontsize=12,
                      figsize=(15, 6), filename=None, title=None,
                      **args):
    """Make a barplot of jackstraw p-values for principal components

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    normalization : `str`
        The name of the normalization to operate on. If this is empty
        or None then the function will be applied on the last
        normalization that was applied.
    fontsize : `int`
        Specifies font size. Default: 12
    figsize : `tuple`
        Figure size in inches. Default: (10, 10)
    linewidth : `float`
        Border width. Default: 0.5
    filename : `str`, optional
        Write to a file instead of showing the plot on screen.
    title : `str`
        Title of the plot. Default: None
    **args
        Passed on into seaborn's violinplot and boxplot functions

    Returns
    -------
    Nothing
    """
    if normalization == None or normalization == '':
        norm = list(obj.norm_data.keys())[-1]
    else:
        norm = normalization
    try:
        target = obj.norm_data[norm]
    except KeyError:
        raise Exception('"%s" not found' % norm)
    try:
        js_p = target['dr']['jackstraw']['results_by_comp']
    except KeyError:
        raise Exception('Run jackstraw first (adobo.dr.jackstraw).')
    plt.clf()
    plt.close(fig='all')
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    bp = sns.barplot(x=np.arange(1, len(js_p.chi2_p)+1), y=-np.log10(js_p.chi2_p),
                     color='black')
    ax.axhline(-np.log10(max(js_p.loc[js_p.significant == True,
               :].chi2_p)), ls='--', color='red')
    bp.set_xlabel("Principal component", fontsize=fontsize)
    bp.set_ylabel("-log10(p-value)", fontsize=fontsize)
    if title:
        p.set_title(title)
    _mpl_finish(filename, **args)
