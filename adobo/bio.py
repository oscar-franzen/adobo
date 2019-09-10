# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions related to biology.
"""

import re
import adobo._log
import adobo.IO
import adobo.preproc

def cell_cycle_train():
    path_pkg = re.sub('/_log.py', '', adobo._log.__file__)
    path_data = path_pkg + '/data/Buettner_2015.mat'
    path_gene_lengths = path_pkg + '/data/Buettner_2015.mat.lengths'
    desc = 'Buettner et al. (2015) doi:10.1038/nbt.3102'
    obj = adobo.IO.load_from_file(path_data, desc=desc)
    adobo.preproc.detect_ercc_spikes(obj, ercc_pattern='NA_ERCC-[0-9]+')
    adobo.normalize.norm(obj, method='rpkm', gene_lengths=path_gene_lengths)
    #path_gene_lengths
    return obj
    
def cell_cycle_predict(obj):
    """Predicts cell cycle phase

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    genes_s_phase : `list`
        A list of S phase genes.
    genes_g2m_phase : `list`
        A list of G2/M phase genes.

    Returns
    -------
    None
    """
    pass
