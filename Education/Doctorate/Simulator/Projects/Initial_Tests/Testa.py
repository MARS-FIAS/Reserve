# Catalyzer

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy import interpolate
import pickle

# My classes!

# from BiochemStem import BiochemStem
# from BiochemSimul import BiochemSimul
# from BiochemAnalysis import BiochemAnalysis

# BiochemStem Start # Control Transcription

# P_ := Promoter Binding Site _
# TF := Transcription Factor
# mRNA

nP = 4 # nP = 4 ~ nTF = 1000
nTF = 2000 # 940
initial_state = {f'P{_}': (1 if _ == 0 else 0) for _ in range(nP+1)}
initial_state.update({'TF': nTF, 'mRNA': 0})

# rates = {'kf': 0.1, 'kb': 1}
u = 5
k0 = 100
pre = 2000 # 1000 # After to 20
rates = {'kf': 0.01, 'kd': pre*0.0017, 'km': pre*0.17/1} # rates = {'kf': 0.5, 'kd': 0.005}
rates.update({f'kb{_}': (k0 if _ == 1 else k0*pow(1/u, _-1)) for _ in range(1, nP+1)})

e = BiochemStem(initial_state, rates)

# Forward

for _ in range(nP):
    prop_fun = f'P{_}*TF*kf'
    delta = {f'P{_}': -1, f'P{_+1}': 1, 'TF': -1}
    e.add_reaction(f'P{_+1}f', prop_fun, delta)

e.add_reaction('mRNAf', f'P{nP}*km', {'mRNA': 1})

# Backward

for _ in range(nP):
    prop_fun = f'P{_+1}*kb{_+1}'
    delta = {f'P{_}': 1, f'P{_+1}': -1, 'TF': 1}
    e.add_reaction(f'P{_}b', prop_fun, delta)

e.add_reaction('mRNAb', 'mRNA*kd', {'mRNA': -1})

# BiochemStem Final # Control Transcription

e.assemble()
e.assembly

########
########







#%%
# nP = 4
NTF = 2000
steps = 200000 # steps = 50000
trajectories = 1
jumps = 100
species = 'mRNA'

pre = 2000 # 2000 # 20
rates = {'kf': 0.01, 'kd': pre*0.0017, 'km': pre*0.17/1} # rates = {'kf': 0.5, 'kd': 0.005}
rates.update({f'kb{_}': (k0 if _ == 1 else k0*pow(1/u, _-1)) for _ in range(1, nP+1)})

# Always check extra variables for simulation/analysis:e.g., I forgot to include kappa once and it gave wrong data!

def simulator(nP, NTF, rates, steps, trajectories, jumps, species, kappa):
    
    d = {} # A container for the activation number!
    
    for nTF in range(0, NTF+1, jumps):
        # if nTF == 0:
        #     nTF = 1
        initial_state = {f'P{_}': (1 if _ == 0 else 0) for _ in range(nP+1)}
        initial_state.update({'TF': nTF, 'mRNA': 0})
        e = BiochemStem(initial_state, rates)
        
        # Forward
        
        for _ in range(nP):
            prop_fun = f'P{_}*TF*kf'
            delta = {f'P{_}': -1, f'P{_+1}': 1, 'TF': -1}
            e.add_reaction(f'P{_+1}f', prop_fun, delta)

        e.add_reaction('mRNAf', f'P{nP}*km', {'mRNA': kappa})
        
        # Backward

        for _ in range(nP):
            prop_fun = f'P{_+1}*kb{_+1}'
            delta = {f'P{_}': 1, f'P{_+1}': -1, 'TF': 1}
            e.add_reaction(f'P{_}b', prop_fun, delta)

        e.add_reaction('mRNAb', 'mRNA*kd', {'mRNA': -1})
        
        e.assemble()
        # e.assembly
        
        w = BiochemSimul(e, steps, trajectories)
        w.meth_direct()
        
        alias = BiochemAnalysis(w)
        
        d.update({nTF: {'Ave': alias.mean(species, 0), 'Var': alias.variance(species, 0)}})
        
        print(nTF)
        
    return d