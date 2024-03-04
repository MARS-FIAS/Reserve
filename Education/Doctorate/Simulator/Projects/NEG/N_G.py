##############################
########              ########
######## NANOG / GATA ########
########              ########
##############################

# Libraries

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import numba
import pandas as pd
import math
from scipy import stats
from scipy import optimize # Curve Fitting
from scipy import interpolate
import pickle
import sys
import gc

#%%# Curve Fitting (Hill Function - Activator / Repressor)

# Closure!

def _hill_fun(h, kind):
    def hill_fun(x, c):
        if kind == 'act':
            return np.power(x, c) / (pow(h, c) + np.power(x, c))
        elif kind == 'rep':
            return pow(h, c) / (pow(h, c) + np.power(x, c))
    return hill_fun

#%%# Biochemical System Definition

# Regulation Motifs

regulations = {'N': {'N': 1, 'E': -1, 'G': -1}, 'E': {'N': 0, 'E': 0, 'G': 0}, 'G': {'N': -1, 'E': 1, 'G': 1}}

# Regulation: positive XOR negative

positives = {X: {Y: np.nan if regulations[X][Y] == 0 else not np.signbit(regulations[X][Y]) for Y in regulations[X].keys()} for X in regulations.keys()}
negatives = {X: {Y: np.nan if regulations[X][Y] == 0 else np.signbit(regulations[X][Y]) for Y in regulations[X].keys()} for X in regulations.keys()}
repressors = negatives

# Species/States

proteins = [X for X in regulations.keys() if not all([Y == 0 for Y in regulations[X].values()])]
promoters = proteins
binding_sites = proteins
coop = 4
cooperativity = [str(_) for _ in range(coop+1)] # Cooperativities?
promoter_binding_sites = [P+S+C for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S]]

# Artificial Species

cases = ['A', 'I'] # {'A': 'Activated', 'I': 'Inactivated'}
arts = [P+_ for P in promoters for _ in cases if not all([not _ for _ in regulations[P].values()])]

#%%# Rate Constants

D = 10e-12 # pow(microns, 2)/seconds # pow(length, 2)/time # Protein diffusion constant
protein_cross_section = 10e-9 # Nanometers
cell_radius = 5e-6 # 10 Micrometers
cell_volume = 4*math.pi*pow(cell_radius, 3)/3 # pow(meters, 3)

pub = 5 # This is only valid for backward_rates
kf_promoter = 4*math.pi*protein_cross_section*D/cell_volume # 3*pow(10, -4)
# kb_promoter = pow(10, 3)*kf_promoter # 1/pow(5, s) # s ~ cooperativity ~ number of binding sites
q = 1 # Important variable!
print('\n\n\tq = ' + str(q) + '\n\n')
kb_promoter = {art: 250*kf_promoter if 'A' in art else 1000*kf_promoter*q if 'N' in art else 1000*kf_promoter/q for art in arts}

_protein_lifetime = (5, 6, 7) # Hours
protein_lifetime = _protein_lifetime[1]*pow(60, 2) # Seconds
protein_copy_number = 100 # Steady-state (old) assumption # kf/kb = 1000

kf_protein = protein_copy_number/protein_lifetime
kb_protein = 1/protein_lifetime
kf_spontaneous = kf_protein/10 # 10*kb_protein # mRNA leakage
kb_spontaneous = kb_protein

#%%# Biochemical System Construction

# Spontaneous Reactions

# Forward

soft_exes = [f'0 -> {P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
soft_props = [f'(1-np.sign({P}I+{P}A))*kf_{P}0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
soft_deltas = [{f'{P}': 1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
soft_rates = {f'kf_{P}0': kf_spontaneous for P in proteins if not all([not _ for _ in regulations[P].values()])}
soft_species = {f'{P}': 0 for P in proteins if not all([not _ for _ in regulations[P].values()])}

# Backward

suba_exes = [f'{P} -> 0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
suba_props = [f'{P}*kb_{P}0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
suba_deltas = [{f'{P}': -1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
suba_rates = {f'kb_{P}0': kb_spontaneous for P in proteins if not all([not _ for _ in regulations[P].values()])}
suba_species = soft_species

# Non-Spontaneous Reactions # Function!

# Forward

forward_exes = [f'{S} + {P}{S}{C} -> {P}{S}'+str(int(C)+1) for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
forward_props = [f'{P}{S}{C}*{S}*kf_{P}{S}'+str(int(C)+1) for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
forward_deltas = [{f'{S}': -1, f'{P}{S}{C}': -1, f'{P}{S}'+str(int(C)+1): 1, P+'I' if repressors[P][S] else P+'A': 1 if int(C)+1 == coop else 0} for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
forward_rates = {f'kf_{P}{S}'+str(int(C)+1): kf_promoter for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop}
forward_species = {f'{P}{S}{C}': 1-np.sign(int(C)) for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S]}

# Backward

backward_exes = [f'{P}{S}'+str(int(C)+1)+f' -> {P}{S}{C} + {S}' for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
backward_props = [f'{P}{S}'+str(int(C)+1)+f'*kb_{P}{S}{C}' for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
backward_deltas = [{f'{S}': 1, f'{P}{S}{C}': 1, f'{P}{S}'+str(int(C)+1): -1, P+'I' if repressors[P][S] else P+'A': -1 if int(C)+1 == coop else 0} for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
backward_rates = {f'kb_{P}{S}{C}': kb_promoter[f'{S}A']/pow(pub, int(C)) if positives[P][S] else kb_promoter[f'{S}I']/pow(pub, int(C)) for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop}
backward_species = forward_species

# Artificial Reactions

art_exes = [f'{P}A -> {P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
art_props = [f'np.sign({P}A)*(1-np.sign({P}I))*kf_{P}1' for P in proteins if not all([not _ for _ in regulations[P].values()])]
art_deltas = [{f'{P}': 1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
art_rates = {f'kf_{P}1': kf_protein for P in proteins if not all([not _ for _ in regulations[P].values()])}
art_species = {art: 0 for art in arts}

#%%# BiochemStem

flags = ['soft', 'suba', 'forward', 'backward', 'art']

initial_state = {}
initial_state.update(soft_species)
initial_state.update(forward_species)
initial_state.update(art_species)

rates = {}
for _ in flags:
    exec(f'rates.update({_}_rates)')

proto = BiochemStem(initial_state, rates)

for flag in flags:
    exec(f'indices = range(len({flag}_exes))')
    for index in indices:
        name = eval(f'{flag}_exes[{index}]')
        prop_fun = eval(f'{flag}_props[{index}]')
        delta = eval(f'{flag}_deltas[{index}]')
        proto.add_reaction(name, prop_fun, delta, verbose = False)

proto.assemble()
# print(proto.assembly)

#%%# BiochemSimul

seed = 52 # 73
trajectories = 250
steps = 250000
press = BiochemSimulMule(proto, steps, trajectories, seed)
%time press.meth_direct()

#%%# BiochemAnalysis

alias = BiochemAnalysis(press)

what = 'nor' # 'hist'
where = (0, steps) # Time slicing
trajectory = 100
safe = False

species = 'N'
alias.plotful(what, where, species, trajectory)
if safe:
    plt.savefig(species+'_'+str(trajectory)+'.jpeg', dpi = 250, quality = 95)
plt.show()
# (seed = 27, trajectory = 50, kf_spontaneous = kf_protein/1, protein_copy_number = 250)
species = 'NI'
alias.plotful(what, where, species, trajectory)
if safe:
    plt.savefig(species+'_'+str(trajectory)+'.jpeg', dpi = 250, quality = 95)
plt.show()

species = 'G'
alias.plotful(what, where, species, trajectory)
if safe:
    plt.savefig(species+'_'+str(trajectory)+'.jpeg', dpi = 250, quality = 95)
plt.show()

species = 'GI'
alias.plotful(what, where, species, trajectory)
if safe:
    plt.savefig(species+'_'+str(trajectory)+'.jpeg', dpi = 250, quality = 95)
plt.show()

#%%# First-Time Passage (Temp Data)

path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Data'

safe = False # Store Temp?
if safe:
    file = open(f'{path}/Temp', 'wb')
    exec(f'pickle.dump(press, file)')
    file.close()

lad = False
if lad:
    file = open(f'{path}/Temp', 'rb')
    exec(f'press = pickle.load(file)')
    file.close()

#%%# First-Time Passage (Prep)

specs = [0, 1]
cut = 0

maxi_pots = np.argmax(press.state_tor[:, specs, :], 0)
maxi = np.max(press.state_tor[:, specs, :], 0)
sieve = np.apply_along_axis(lambda x: x >= cut, 0, maxi)
_con = np.invert(sieve)
con = np.apply_along_axis(lambda x: x[0]*x[1], 0, _con)

alps = maxi[0][sieve[0]] # NANOG
_alps = maxi[1][sieve[0]]
print('NANOG', alps.shape, '\n')
bets = maxi[1][sieve[1]] # GATA
_bets = maxi[0][sieve[1]]
print('GATA', bets.shape, '\n')
cars = maxi[0][con] # Rest
_cars = maxi[1][con]
print('Rest', cars.shape, '\n')

start = 0
end = 100

lis = ['alps', 'bets', 'cars']
for _ in lis:
    temp = eval(_)
    _temp = eval(f'_{_}')
    plt.title(_)
    plt.plot(temp[start:end])
    plt.plot(_temp[start:end])
    plt.show()

# Experimental! Start!
_lis = np.array(range(trajectories))
lis = _lis[sieve[0]]
print(np.argwhere(_alps == 149))
# Experimental! End!

#%%# First-Time Passage (Analysis)

specs = [0, 1]
cut = 500
value = 2*cut

value_pots = np.argwhere(press.state_tor[:, specs, :] == 1000)

_value_pots = np.argwhere(press.state_tor[:, 1, :] == 1000)
maxi = np.max(press.state_tor[:, specs, :], 0)
sieve = np.apply_along_axis(lambda x: x >= cut, 0, maxi)
_con = np.invert(sieve)
con = np.apply_along_axis(lambda x: x[0]*x[1], 0, _con)

alps = maxi[0][sieve[0]] # NANOG
_alps = maxi[1][sieve[0]]
# alps_ties = press.epoch_mat[maxi_pots[0][sieve[0]], :]
print('NANOG', alps.shape, '\n')
bets = maxi[1][sieve[1]] # GATA
_bets = maxi[0][sieve[1]]
print('GATA', bets.shape, '\n')
cars = maxi[0][con] # Rest
_cars = maxi[1][con]
print('Rest', cars.shape, '\n')

lis = ['alps', 'bets', 'cars']
for _ in lis:
    temp = eval(_)
    _temp = eval(f'_{_}')
    plt.title(_)
    plt.plot(temp)
    plt.plot(_temp)
    plt.show()
