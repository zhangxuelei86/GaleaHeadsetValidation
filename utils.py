import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np


def get_n_distinct_colors(n_colors):
    cm = plt.get_cmap('gist_rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=n_colors - 1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    return [scalarMap.to_rgba(i) for i in range(n_colors)]



from math import floor


def conv1d_output_shape(s, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    :rtype: object
    """
    s_out = floor(((s + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
    return s_out
