#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from python.poisson import PoissonControl
from python.Laminar import Laminar
#exec(open("/home/nwycoff_umass_edu/backup/tmp/python/Laminar.py").read())
from python.KiriE import ShellEnergy
from python.settings import *
from dolfin import *
import pickle
from python.active_lib import sample_m
from glob import glob
import numpy as np

import json
from ufl import (
    FiniteElement, VectorElement, TensorElement, MixedElement
)
try:
    from ufl import EnrichedElement
except Exception:
    EnrichedElement = None  # some builds omit it

funcs = ['poisson', 'laminar','kiri']
#comps = ['vanil','asm']
comps = ['asm','vanil']


control_objs = {}
samplers = {}

def get_R(func):
    if func=='kiri':
        R = 10
    elif func=='poisson':
        R = 2
    elif func=='laminar':
        #R = 1
        #R = 2
        R = 4
    else:
        raise Exception("Please visually determine active subspace size for unknown function.")
    return R

for func in funcs:
    if func=='poisson':
        control_objs[func] = lambda **kwargs: PoissonControl(nx=32, ny=32, alpha=1e-3, **kwargs)
        samplers[func] = lambda co, **kwargs: sample_m(co.Q)
    elif func=='laminar':
        control_objs[func] = lambda **kwargs: Laminar(unit=.1, nx=40, ny=40, nu=4.0e-2, stokes=False, **kwargs)
        samplers[func] = lambda co, **kwargs: co.sample_inflow_KL(**kwargs)[0]
        print("Smol sigma in sampling.")
    elif func=='kiri':
        control_objs[func] = lambda **kwargs: ShellEnergy(xdmf_path="data/only_spiral.xdmf", use_spiral_perim=True, **kwargs)
        samplers[func] = lambda co, **kwargs: co.sample_m()
    else:
        raise Exception("Unknown func!")

def rebuild_u_on_master(co, u_vecs):
    """
    u_vecs: list of 1D numpy arrays (coefficient vectors from workers)
    Returns: list of Function(self.Uspace) built on a single master energy
    """

    #energy = ShellEnergy(xdmf_path=mesh_path, use_spiral_perim=use_spiral_perim)
    #energy._ensure_canonical_spaces()
    V = co.Q

    out = []
    for arr in u_vecs:
        if arr.shape[0] != V.dim():
            raise ValueError(f"Length mismatch: got {arr.shape[0]} but Uspace.dim()={V.dim()}")

        u_fun = Function(V)
        vec = u_fun.vector()
        vec.set_local(arr)       # O(N) copy into the PETSc/DOLFIN vector
        vec.apply("insert")      # finalize
        out.append(u_fun)

    return out

## Reconstitute output values.
def load_data(co, func, eig=True):
    fM = []
    m_vecs = []
    g_vecs = []
    for fname in sorted(glob(f'{dumpdir}/fM_{func}_*.pkl')):
        with open(fname,'rb') as f:
            fM_i, m_vecs_i, g_vecs_i = pickle.load(f)
        fM.extend(fM_i)
        m_vecs.extend(m_vecs_i)
        g_vecs.extend(g_vecs_i)
        funcs = m_vecs + g_vecs

    lens = [len(m_vecs), len(g_vecs)]

    if eig:
        with open(eigfile(func),'rb') as f:
            OMEGA, GAMMA, ed, e_vecs = pickle.load(f)
        funcs += e_vecs
        #lens += [len(e_vecs)]

    lens = np.cumsum(lens)


    ## Reconstitute input functions
    all_funs = rebuild_u_on_master(co, funcs)
    #assert (len(all_funs) % pks) == 0
    M = all_funs[:lens[0]]
    G = all_funs[lens[0]:lens[1]]
    if eig:
        E = all_funs[lens[1]:]
    B = len(fM)
    assert B == len(M)
    assert B == len(G)
    if eig:
        #assert B == len(E)
        return M,G,fM,B,GAMMA,OMEGA,ed,E
    else:
        return M,G,fM,B
