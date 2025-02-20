import numpy as np
import matplotlib as mpl

def latexify(columns=2):
  """
  Set up matplotlib's RC params for LaTeX plotting.
  Call this before plotting a figure.
  Ref: https://matplotlib.org/tutorials/introductory/customizing.html
  Parameters
  ----------
  columns : {1, 2}
  """

  assert(columns in [1,2])

  fig_width_pt  = 240.0 if (columns == 1) else 504.0
  inches_per_pt = 1.0/72.27                    # Convert pt to inch
  golden_mean   = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
  fig_width     = fig_width_pt*inches_per_pt   # width in inches
  fig_height    = fig_width*golden_mean*1.3    # height in inches
  fig_size      = [fig_width,fig_height]

  params = {'pdf.fonttype': 42,
            'ps.fonttype': 42,
            # 'font.family': 'sans serif',
            # 'font.sans-serif': 'Helvetica',
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'text.usetex': False,
            'figure.figsize': fig_size,
            'figure.titlesize' : 12,
            'figure.dpi' : 300,
            'axes.prop_cycle' : mpl.cycler('color', ['#4e79a7', '#f28e2b', \
                                                     '#e15759', '#76b7b2', \
                                                     '#59a14f', '#edc948', \
                                                     '#b07aa1', '#ff9da7', \
                                                     '#9c755f', '#bab0ac'])
            }

  mpl.style.use('default')
  mpl.rcParams.update(params)