def normalize(obj, method='depth', log=True, remove_low_qual_cells=True,
              exon_lengths=None):
    """Normalize gene expression data

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : {'depth', 'rpkm'}
        Specifies the method to use. `depth` refers to simple normalization strategy
        involving dividing by total number of reads per cell. `rpkm` performs RPKM
        normalization and requires the `exon_lengths` parameter to be set.
    log : `bool`
        Perform log2 transformation (default: True)
    remove_low_qual_cells : `bool`
        Remove low quality cells identified using :py:meth:`adobo.preproc.find_low_quality_cells`.
    exon_lengths : :class:`pandas.DataFrame`
        A pandas data frame containing two columns; first column should be gene names
        matching the data matrix and the second column should contain exon lengths.

    Returns
    -------
    None
    """
    data_cp = data.copy()
    
    if remove_low_quality:
        data_cp = data_cp.drop(self.low_quality_cells, axis=1)

    if not mrnafull and input_type == 'raw':
        col_sums = data_cp.apply(lambda x: sum(x), axis=0)
        data_norm = (data_cp / col_sums) * 10000
        data_norm = np.log2(data_norm+1)
    elif mrnafull and input_type == 'raw':
        data_norm = self.rpkm(data_cp)
    elif input_type == 'rpkm':
        log_debug('normalization() Running log2')
        data_norm = np.log2(data_cp+1)
    else:
        data_norm = data_cp
        log_debug('Normalization is not needed.')

    if fn_out != '':
        self._dump(data_norm, fn_out, compress=True)

    log_debug('Finished normalize()')
