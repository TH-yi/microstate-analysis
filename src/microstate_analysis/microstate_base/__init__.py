import os
import numpy as np
from scipy import signal, stats, spatial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from microstate_analysis.eeg_tool.math_utilis import zero_mean, condensed_to_square
from microstate_analysis.eeg_tool.utilis import read_subject_info, read_info, write_info
import codecs
import json
from collections import OrderedDict
import mne
from multiprocessing import Pool
import itertools
from operator import itemgetter, attrgetter
from statsmodels.stats.multitest import multipletests
from sklearn import preprocessing

import importlib
import pkgutil

from microstate_analysis.microstate_base.microstate import Microstate
from microstate_analysis.microstate_base.meanmicrostate import MeanMicrostate

for module_info in pkgutil.iter_modules(__path__):
    module_name = module_info.name
    module = importlib.import_module(f".{module_name}", package=__name__)
    globals()[module_name] = module

__all__ = [
    'os', 'np', 'signal', 'plt', 'stats', 'spatial', 'zero_mean',
    'condensed_to_square', 'read_subject_info', 'read_info', 'codecs',
    'json', 'OrderedDict', 'mne', 'Pool', 'gridspec', 'write_info',
    'itertools', 'itemgetter', 'attrgetter', 'multipletests', 'preprocessing',
    'Microstate', 'MeanMicrostate'
]


__all__.extend([name for name in globals() if not name.startswith('_')])
