adobo for scRNA-seq

# Launch Python
```bash
python3
```

# Basic usage - loading a dataset of raw read counts
```python
import adobo
# create a new data object
d = adobo.data()
d.load_from_file('GSE95315_10X_expression_data.brain.tab', verbose=True, column_id=True)

# remove empty cells and genes
d.remove_empty(verbose=True)

# remove mitochondrial genes
d.remove_mito(verbose=True)

# detect ERCC spikes
d.detect_ERCC_spikes(verbose=True)
```

# Quality control
```python
# barplot of number of reads per cell
d.barplot_reads_per_cell()
```
