#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

#B_each = 4
#B_each = 4
B_each = 100
#B_each = 1
#B = 400
B = 10000
#B = 100
#B = 100
#B = 10

dumpdir = '/work/pi_nwycoff_umass_edu/aistats_dump/'

os.makedirs(dumpdir, exist_ok=True)
fMfile = lambda sig: f"{dumpdir}/fM_{sig}.pkl"
EOfile = lambda sig: f"{dumpdir}/eig_other_{sig}.pkl"
EFfile = lambda sig: f"{dumpdir}/eig_functions_{sig}"

meshfile = lambda sig: f"mesh/{sig}_mesh.xdmf"

#allfile = lambda sig: f"dump/all_{sig}.h5"
eigfile = lambda sig: f"{dumpdir}/eig_{sig}.pkl"

sim_out = lambda sig: f"sim_out/{sig}/"
debug_path = lambda sig: f"debug/{sig}/"


## BO settings.
