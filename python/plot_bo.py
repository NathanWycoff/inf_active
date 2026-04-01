#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "/home/nwycoff_umass_edu/inf_act/")

import pickle
from python.settings import *
from python.common import comps
import glob
import numpy as np
import matplotlib.pyplot as plt

do_log = ['poisson','laminar','kiri']

if hasattr(sys, 'ps1'):
    assert len(sys.argv)==1
    #func='kiri'
    #func='laminar'
    func='poisson'
else:
    assert len(sys.argv)==2
    func = sys.argv[1]
    
print(func)

###### ###### ###### ###### ###### ###### ###### ######
# Read in data.
###### ###### ###### ###### ###### ###### ###### ######
files = glob.glob(sim_out(func)+f'bo_{func}_*')
print(f"Found {len(files)} files.")
#print(files)

perfs = dict([(c,[]) for c in comps])
for fn in files:
    with open(fn,'rb') as f:
        perf = pickle.load(f)
    for c in comps:
        perfs[c].append(np.minimum.accumulate(perf[c]))

for c in comps:
    perfs[c] = np.stack(perfs[c])
    

###### ###### ###### ###### ###### ###### ###### ######
# Compute stats.
###### ###### ###### ###### ###### ###### ###### ######
medians = {}
lb = {}
ub = {}
for c in comps:
    medians[c] = np.median(perfs[c],axis=0)
    lb[c] = np.quantile(perfs[c],0.1,axis=0)
    ub[c] = np.quantile(perfs[c],0.9,axis=0)

###### ###### ###### ###### ###### ###### ###### ######
# Graphics
###### ###### ###### ###### ###### ###### ###### ######
cols = {'vanil':'tab:blue', 'asm':'tab:orange'}

pretty_names = {
        'vanil' : 'Rand',
        'asm' : 'ASM'
        }

#def strip_leading_zero(y, _):
#    # format nicely, then remove leading "0."
#    s = f"{y:g}"
#    if s.startswith("0.") and not s.startswith("0.0"):  # e.g. 0.2 → .2
#        return s[1:]
#    elif s.startswith("-0."):
#        return "-" + s[2:]  # -0.3 → -.3
#    return s

import numpy as np
import matplotlib.ticker as mticker

def format_y_axis_universal(ax, offset_x=-0.15, offset_fs=6, decimals=2):
    """Uniform y-axis formatting for both linear and log scales, robust for <1-decade spans."""
    if ax.get_yscale() == 'log':
        ymin, ymax = ax.get_ylim()
        # Guard: if limits invalid for log, bail out gracefully
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin <= 0 or ymax <= 0:
            return

        span_decades = np.log10(ymax) - np.log10(ymin)

        # Choose a representative global exponent k (geometric-mean decade)
        k = int(np.floor(0.5 * (np.log10(ymin) + np.log10(ymax))))
        scale = 10.0 ** k
        fmt = "{:." + str(decimals) + "f}"

        if span_decades < 1.0:
            # If we don't cross a full decade, promote 2–9×10^n to MAJOR ticks
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=range(1,10), numticks=15))
            ax.yaxis.set_minor_locator(mticker.NullLocator())  # keep it clean
        else:
            # Normal: majors at decades, minors at 2–9
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2,10), numticks=10))

        # Label majors with scaled two-decimal numbers
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: fmt.format(y / scale)))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # Draw/position the “×10^k” offset text (left over tick labels)
        off = ax.yaxis.get_offset_text()
        off.set_text(rf"$\times 10^{{{k}}}$")
        off.set_x(offset_x)
        off.set_fontsize(offset_fs)

    else:
        # Linear: ScalarFormatter with scientific multiplier
        sf = mticker.ScalarFormatter(useMathText=True)
        sf.set_scientific(True)
        sf.set_powerlimits((-2, 2))
        sf.set_useOffset(False)
        ax.yaxis.set_major_formatter(sf)

        off = ax.yaxis.get_offset_text()
        off.set_x(offset_x)
        off.set_fontsize(offset_fs)

print("overwriting comps!")
comps = ['vanil','asm']
import matplotlib.ticker as mticker
fig = plt.figure(figsize=[2,2])
for c in comps:
    plt.plot(medians[c], label = pretty_names[c], color = cols[c])
    plt.plot(lb[c], color = cols[c], linestyle='--')
    plt.plot(ub[c], color = cols[c], linestyle='--')
if func in do_log:
    plt.yscale('log')
#if func=='poisson':
if True:
    plt.legend(loc='lower left', prop = {'weight' : 'bold'}, frameon=False, handlelength=1.0, handletextpad=0.2, bbox_to_anchor=(-0.05,0.))
plt.xlabel("Budget", fontdict = {'weight' : 'bold'}, labelpad=-7)
plt.ylabel("BOV", fontdict = {'weight' : 'bold'}, labelpad=1)
plt.title(func, fontdict = {'weight' : 'bold'})
# Major ticks at powers of 10; label them as plain numbers
ax = plt.gca()
format_y_axis_universal(ax, offset_x=-0.15, offset_fs=6, decimals=2)
plt.yticks(fontsize=6)
plt.xticks(fontsize=8)
# Reduce tick padding
ax.tick_params(axis='y', which='major', pad=0.2)
ax.tick_params(axis='x', which='major', pad=0.2)
if func=='poisson':
    ax.set_xticks([0,20])
else:
    ax.set_xticks([0,50])
plt.tight_layout()
plt.savefig(f"bo_{func}.pdf")
plt.close()

