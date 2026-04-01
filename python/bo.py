#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ax.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.25.2025

import sys
sys.path.insert(0, "/home/nwycoff_umass_edu/inf_act/")

import numpy as np
from ax import Client, RangeParameterConfig
from dolfin import *
import numpy as np
from python.active_lib import gram_matrix, dist_matrix, sample_m, get_eigenfuncs, linear_combination, pull_to_mesh
from python.common import *
import matplotlib.pyplot as plt
import pickle
from python.settings import *
import os
from python.KiriE import silence_everything

if hasattr(sys, 'ps1'):
    assert len(sys.argv)==1
    seed = 101
    #func='kiri'
    func='laminar'
    #func='poisson'
else:
    assert len(sys.argv)==3
    func = sys.argv[1]
    seed = int(sys.argv[2])
    
print(func)
print(seed)
np.random.seed(seed)

###############
## Load data.
co = control_objs[func]()
M,G,fM,B,_,OMEGA,ed,E = load_data(co,func)
fM = np.array(fM)

print(f"B={B}")

NA_VAL = np.max(fM)

N_init = 10
if N_init < B:
    print("Not using all preruns for init!")
inds_start = np.random.choice(B, N_init, replace = False)
#budget = 40 if func=='kiri' else 10
if func=='kiri':
    budget = 40
elif func=='laminar':
    budget = 40
else:
    budget = 10

#P = 2
P = get_R(func)
#P = 5
#P = B

###### ###### ###### ###### ###### ###### ###### ######
# Build objective functions
###### ###### ###### ###### ###### ###### ###### ######
#if func=='kiri':
if True:
    co_M = control_objs[func](mesh=M[0].function_space().mesh())
    co_E = control_objs[func](mesh=E[0].function_space().mesh())
else:
    co_M = co_E = control_objs[func]()
# Sanity-check: ensure output of first m matches stored data.
#with silence_everything():
#    m_on_coQ = project(M[0], co_M.Q)
J_val, g = co_M.evaluate(M[0])
print(J_val)
if abs(J_val - fM[0]) != 0:
    print("Warning! Nonzero difference in control function:")
    print(J_val - fM[0])
else:
    print("Reproduced first eval exactly.")

#####
## BO
## NOTE: E is in the default numpy ASCENDING order, i.e. E[-1] is the leading eigenvector!
## NOTE: However, xi is in the non-insane, rational, non-aneurysm-inducing DESCENDING order
## NOTE: Thus, we index E in reverse.
def obj_ASM(xi):
    coef = np.array([xi[f"x{p}"] for p in range(P)])
    print(coef)

    #g_lin = linear_combination(np.flip(E[-P:]), coef)
    g_lin = linear_combination(E[-P:][::-1], coef)

    print(g_lin)

    #with silence_everything():
    #    g_lin_on_coQ = project(g_lin, co_E.Q)
    #J_val, _ = co_E.evaluate(g_lin_on_coQ)
    J_val, _ = co_E.evaluate(g_lin)
    print(J_val)

    return J_val

#####
## Vanil
print("DRY violation in objectives!")
def obj_VANIL(xi):
    coef = np.array([xi[f"x{p}"] for p in range(P)])
    print(coef)

    g_lin = linear_combination(U, coef)
    print(g_lin)

    #with silence_everything():
    #    g_lin_on_coQ = project(g_lin, co_M.Q)
    #J_val, _ = co_M.evaluate(g_lin_on_coQ)
    J_val, _ = co_M.evaluate(g_lin)
    print(J_val)

    return J_val

objs = {
        'asm': obj_ASM,
        'vanil' : obj_VANIL
        }
Xi_dicts = {}
y_inits = {}
coords = {}

###### ###### ###### ###### ###### ###### ###### ######
# Resolve Initial Design Coordinates
###### ###### ###### ###### ###### ###### ###### ######
#####
## Vanil
u_inds = np.random.choice(B,P,replace=False)
U = [M[ui] for ui in u_inds]
coords['vanil'] = np.linalg.solve(gram_matrix(U,U), gram_matrix(U,M)).T
X_init = coords['vanil'][inds_start,:]

# DRY2
Xi_dicts['vanil'] = [dict([(f"x{p}",X_init[n,p]) for p in range(P)]) for n in range(X_init.shape[0])]
y_inits['vanil'] = np.array([float(objs['vanil'](xi)) for xi in Xi_dicts['vanil']])

#####
## BO
coords['asm'] = OMEGA
print("TODO: ASM coords object should already truncate to P so as to match vanil behavior.")
X_init = np.flip(coords['asm'][inds_start,-P:],axis=1)

# DRY2
Xi_dicts['asm'] = [dict([(f"x{p}",X_init[n,p]) for p in range(P)]) for n in range(X_init.shape[0])]
y_inits['asm'] = np.array([float(objs['asm'](xi)) for xi in Xi_dicts['asm']])

rads = {}
for comp in comps:
    print('--')
    print(f"Sanity check for: -- {comp} --")
    print('--')
    print("Correlation between original fM and recomputation.")
    print(np.corrcoef(y_inits[comp], fM[inds_start])[0,1])

    print("Empirical extent along basis:")
    print(f"{np.min(coords[comp])} to {np.max(coords[comp])}")
    rads[comp] = 1.5*np.max(np.abs(coords[comp]))

###### ###### ###### ###### ###### ###### ###### ######
# Run actual BO loop.
###### ###### ###### ###### ###### ###### ###### ######
perf = {}
evals = {}
for comp in comps:
    client = Client()
    client.configure_experiment(
        name=func,
        parameters=[
            RangeParameterConfig(
                name=f"x{p}",
                bounds=(-rads[comp],rads[comp]),
                parameter_type="float",
            ) for p in range(P)],
    )
    client.configure_optimization(objective="-1 * obj")

    init_data = [(Xi_dicts[comp][n], {'obj':y_inits[comp][n]}) for n in range(N_init)]

    for parameters, raw_data in init_data:
        # First attach the trial and note the trial index
        trial_index = client.attach_trial(parameters=parameters)
        client.complete_trial(trial_index=trial_index, raw_data=raw_data)

    for _ in range(budget):
        print('----------------------------------')
        print(f"{comp} - {trial_index}")
        print('----------------------------------')
        for trial_index, parameters in client.get_next_trials(max_trials=1).items():
            try:
                result = objs[comp](parameters)
            except Exception as e:
                print("Caught exception; inputting bad value.")
                print(e)
                result = NA_VAL
            client.complete_trial(
                trial_index=trial_index,
                raw_data={
                    "obj": result
                },
            )

    perf[comp] = client.summarize()['obj']
    evals[comp] = client.summarize()[[f"x{p}" for p in range(P)]]

print(perf)
print([(c,np.min(perf[c])) for c in comps])

print("did you really think you were going to get away with not plotting the perfs?")

os.makedirs(sim_out(func), exist_ok=True)
with open(sim_out(func)+f"bo_{func}_{seed}.pkl", 'wb') as f:
    pickle.dump(perf, f)
