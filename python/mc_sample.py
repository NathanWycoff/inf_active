#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "/home/nwycoff_umass_edu/inf_act/")

import numpy as np
from dolfin import *
from python.active_lib import sample_m, linear_combination, gram_matrix
import pickle
from tqdm import tqdm
from python.common import *
from python.settings import *

if hasattr(sys, 'ps1'):
    assert len(sys.argv)==1
    func = 'laminar'
    #func = 'poisson'
else:
    assert len(sys.argv)==2
    func = sys.argv[1]
    
np.random.seed(123)

n_photos = 10
G_path = debug_path(func) + f'/G/'
M_path = debug_path(func) + f'/M/'
os.makedirs(G_path, exist_ok=True)
os.makedirs(M_path, exist_ok=True)

fM, G, M = [], [], []

co = control_objs[func]()

fails = 0

for b in tqdm(range(B)):
    m = samplers[func](co)
    try:
        J_val, g = co.evaluate(m)
        M.append(m)
        fM.append(J_val)
        G.append(g)
        #print(J_val)

        if b < n_photos:
            co.plot(m, fname=M_path + f"{func}_m_{b}.png")
            co.plot(g, fname=G_path + f"{func}_g_{b}.png")
    except RuntimeError as e:
        print("Skipping...")
        print(e)
        fails += 1

fM = np.array(fM)

print(f"Percent failed runs: {fails/B}")

print("First few J values:", fM[:min(5, B)])

save_functions_xdmf(M, path =Mfile(func))
save_functions_xdmf(G, path =Mfile(func))
# Get rid of bad grads on laminar.
#if func=='laminar':
#    GAMMA = gram_matrix(G)
#    norms = np.sqrt(np.diag(GAMMA))
#    thresh = 1e-2
#    isbig = np.where(norms>thresh)[0]
#    #print(f"Shrinking {len(isbig)} grads...")
#    M = [g for i,g in enumerate(M) if not i in isbig]
#    G = [g for i,g in enumerate(G) if not i in isbig]
#    fM = np.array([g for i,g in enumerate(fM) if not i in isbig])
#    #for i in isbig:
#    #    G[i] = linear_combination([G[i]], np.array([thresh/norms[i]]))
#    #GAMMA = gram_matrix(G)
#    #norms = np.sqrt(np.diag(GAMMA))
#    #co.plot(G[np.argmax(norms)], fname = 'big_boi.png')
#    #norms = np.sqrt(np.diag(GAMMA))
#    #pass

#save_functions_hdf5(M, path=Mfile(func))
#save_functions_hdf5(G, path=Gfile(func))

with open(fMfile(func),'wb') as f:
    pickle.dump(fM, f)


