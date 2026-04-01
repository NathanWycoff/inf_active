#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "/home/nwycoff_umass_edu/inf_act/")

from dolfin import *
import numpy as np
from python.active_lib import gram_matrix, dist_matrix, sample_m, get_eigenfuncs, linear_combination
from python.common import *
import matplotlib.pyplot as plt
import pickle
from python.settings import *
from tqdm import tqdm

if hasattr(sys, 'ps1'):
    assert len(sys.argv)==1
    #func = 'laminar'
    func = 'kiri'
else:
    assert len(sys.argv)==2
    func = sys.argv[1]

co = control_objs[func]()

## Load data.
#M, G, fM, B = load_data(func)
M, G, fM, B = load_data(co, func, eig=False)

print(f'eigenanalysis - {func}')
print(f"B={B}")

#Br = 1000
Br = 5
#Br = 500
if B > Br:
    nsub = 30
    Tl = int(np.sqrt(Br))
    Es = []
    print("Subbag estimator.")
    print(f"{Tl*nsub} eventual functions...")
    for s in tqdm(range(nsub)):
        samp = np.random.choice(B,Br,replace=False)
        Gs = [g for i,g in enumerate(G) if i in samp]

        GAMMA = gram_matrix(Gs)
        E, ed = get_eigenfuncs(GAMMA, Gs)

        for b in range(1,Tl+1):
            Es.append(linear_combination([E[-b]], [ed[0][-b]/np.max(ed[0])]))

    GAMMA = gram_matrix(Es)
    E, ed = get_eigenfuncs(GAMMA, Es)

else:
    GAMMA = gram_matrix(G)
    E, ed = get_eigenfuncs(GAMMA, G)

## Create active subspace dist.
# Step 1: Get coefficients of projected input functions in eigenbasis
OMEGA = gram_matrix(M, E)

#################
## Plots 

R = get_R(func)

## Spectrum.
emax = 20 if func=='kiri' else 10
#emax = 20 
evp = np.min([B,emax])

fig = plt.figure(figsize=[4,4])
plt.scatter(np.arange(evp), np.log10(np.flip(ed[0])[:evp]))
#ax = plt.gca().twinx()
#percent_explained = np.cumsum(np.flip(ed[0]))/np.sum(ed[0])
#ax.plot(np.arange(evp), percent_explained[:evp], color = 'gray', linestyle='--')
plt.xlabel("Index", fontdict = {'weight':'bold'})
plt.ylabel("Eigenvalue", fontdict = {'weight':'bold'})
plt.tight_layout()
plt.savefig(f"evals_{func}.pdf")
plt.close()

## Functions
#scale = np.power(np.mean(np.log10(np.abs(OMEGA))),10.)
for b in range(1,3+1):
    scale = ed[0][-b]
    Eshrunk = linear_combination([E[-b]], np.array([scale]))
    co.plot(Eshrunk, fname = f'{func}_efunc_{b}.pdf')

## 2D plot
Z = np.flip(OMEGA[:,-2:], axis = 1)
#Z = (Z - np.mean(Z,axis=0)[None,:])/np.std(Z,axis=0)[None,:]
ub = np.max(Z,axis=0)
lb = np.min(Z,axis=0)
Z = (Z-lb[None,:]) / (ub-lb)[None,:]

y = np.log10(fM)

gpfit = False
if gpfit:
    from hetgpy import hetGP
    # model
    model = hetGP()
    model.mleHetGP(
       X = Z,
       Z = y,
       covtype = "Gaussian",
       maxit = 100
    )
    ng = 20
    zg = np.linspace(0,1,num=ng)
    Zg = np.stack([np.repeat(zg,ng), np.tile(zg,ng)],axis=1)
    pred = model.predict(Zg)['mean']

fig = plt.figure(figsize=[4,4])
if gpfit:
    plt.tricontourf(Zg[:,0], Zg[:,1], pred)
    plt.colorbar()
    plt.scatter(Z[:,0], Z[:,1], c='white',s=1.1*plt.rcParams['lines.markersize'] ** 2, alpha = 0.1)
else:
    plt.scatter(Z[:,0], Z[:,1], c=y, alpha = 0.7)
plt.savefig(f"{func}_2d.pdf")
plt.close()

e_vecs = [e.vector().get_local().copy() for e in E]
with open(eigfile(func),'wb') as f:
    pickle.dump([OMEGA, GAMMA, ed, e_vecs], f)


