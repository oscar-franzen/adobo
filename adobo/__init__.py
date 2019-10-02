# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
import sys
import re
import os

# prevent unexpected multithreading in numpy
# don't move this. Environmental variables must be set _before_ loading numpy.
os.system('export OPENBLAS_NUM_THREADS=1')
os.system('export MKL_NUM_THREADS=1')
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from .data import dataset
from . import IO
from . import preproc
from . import plotting
from . import normalize
from . import hvg
from . import dr
from . import bio
from . import clustering

debug = 0

# package metadata
__author__ = 'Oscar Franzén'
__email__ = 'p.oscar.franzen@gmail.com'
__url__ = 'https://oscar-franzen.github.io/adobo/'

path = os.path.abspath(os.path.dirname(__file__)) + '/VERSION'
__version__ = open(path).read().replace('\n', '')

# color exceptions in the terminal
def excepthook(type, value, tb):
    import traceback
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import TerminalFormatter

    tbtext = ''.join(traceback.format_exception(type, value, tb))
    lexer = get_lexer_by_name("pytb", stripall=True)
    formatter = TerminalFormatter()
    if debug == 0:
        if re.search('\nException: .+',tbtext):
            tbtext = re.search('Exception: (.+)',tbtext).group(1)
    sys.stderr.write(highlight(tbtext, lexer, formatter))

sys.excepthook = excepthook

print('adobo version %s. Documentation: %s' % (__version__, __url__))
