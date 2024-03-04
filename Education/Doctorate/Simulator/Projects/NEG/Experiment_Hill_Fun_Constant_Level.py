####################################################
########                                    ########
######## Experiment Hill Fun Constant Level ########
########                                    ########
####################################################

#%%# Libraries

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy import optimize # Curve Fitting
from scipy import interpolate
import pickle
import sys
import gc

#%%# Curve Fitting (Hill Function - Activator)

# Closure!

def _hill_fun(h):
    def hill_fun(x, c):
        return np.power(x, c) / (pow(h, c) + np.power(x, c))
    return hill_fun

#%%# Biochemical System Definition

# Regulation Motifs

regulations = {'N': {'N': True}}

# Regulation: positive XOR negative

positives = {'N': {'N': True}}
# negatives = {'N': {'N': False}}
repressors = {'N': {'N': False}}

# Species/States

proteins = ['N']
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
kb_promoter = pow(10, 3)*kf_promoter # 1/pow(5, s) # s ~ cooperativity ~ number of binding sites

_protein_lifetime = (5, 6, 7) # Hours
protein_lifetime = _protein_lifetime[1]*pow(60, 2) # Seconds
protein_copy_number = 100 # Steady-state assumption # kf/kb = 1000

kf_protein = protein_copy_number/protein_lifetime
kb_protein = 1/protein_lifetime
kf_spontaneous = kf_protein/10 # 10*kb_protein # mRNA leakage
kb_spontaneous = kb_protein

#%%# Biochemical System Construction

# Spontaneous Reactions

# Forward

soft_exes = [f'0 -> {P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
soft_props = [f'0*kf_{P}0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
soft_deltas = [{f'{P}': 0} for P in proteins if not all([not _ for _ in regulations[P].values()])]
soft_rates = {f'kf_{P}0': kf_spontaneous for P in proteins if not all([not _ for _ in regulations[P].values()])}
soft_species = {f'{P}': 50 for P in proteins if not all([not _ for _ in regulations[P].values()])}

# Backward

suba_exes = [f'{P} -> 0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
suba_props = [f'0*{P}*kb_{P}0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
suba_deltas = [{f'{P}': 0} for P in proteins if not all([not _ for _ in regulations[P].values()])]
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
backward_rates = {f'kb_{P}{S}{C}': kb_promoter/pow(pub, int(C)) for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop}
backward_species = forward_species

# Artificial Reactions

art_exes = [f'{P}A -> {P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
art_props = [f'0*np.sign({P}A)*(1-np.sign({P}I))*kf_{P}1' for P in proteins if not all([not _ for _ in regulations[P].values()])]
art_deltas = [{f'{P}': 0} for P in proteins if not all([not _ for _ in regulations[P].values()])]
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
print(proto.assembly)

#%%# BiochemSimul

seed = 27
trajectories = 1000
steps = 1000
press = BiochemSimulMule(proto, steps, trajectories, seed)
%time press.meth_direct()

#%%# BiochemAnalysis

alias = BiochemAnalysis(press)

what = 'nor' # 'hist'
where = (0, 100000) # Time slicing
trajectory = 700
safe = False

species = 'N'
alias.plotful(what, where, species, trajectory)
if safe:
    plt.savefig(species+'_'+str(trajectory)+'.jpeg', dpi = 250, quality = 95)
plt.show()

species = 'NA'
alias.plotful(what, where, species, trajectory)
if safe:
    plt.savefig(species+'_'+str(trajectory)+'.jpeg', dpi = 250, quality = 95)

#%%# BiochemSimulMule

seed = 25
trajectories = 1000
steps = 10000
mule = 50
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Data'

coves = range(coop, mule*coop+1, coop)
dit = {0: (0, 0)} # Relative Time ON
dot = {0: (np.nan, np.nan)} # Simulation Time

for cove in coves:
    proto.initial_state['N'] = cove
    print('N Copy Number:\t', cove)
    press = BiochemSimulMule(proto, steps, trajectories, seed)
    press.meth_direct()
    testa = BiochemAnalysis(press)
    _mean_NA = [testa.mean('NA', _) for _ in range(trajectories)]
    mean_NA = np.mean(_mean_NA)
    stan_err_NA = np.std(_mean_NA)
    print('\tSample Mean:', mean_NA, '\tStandard Error (Mean):', stan_err_NA)
    dit.update({cove: (mean_NA, stan_err_NA)})
    print('\n')
    mean_time = np.mean(np.max(press.epoch_mat/pow(60, 2), 0))
    stan_deva_time = np.std(np.max(press.epoch_mat/pow(60, 2), 0))
    dot.update({cove: (mean_time, stan_deva_time)})

safe = False # Store Statistics?

if safe:
    # Dit
    file = open(f'{path}/Ex_Two_{seed}_{trajectories}_{steps}_{mule}', 'wb')
    exec(f'pickle.dump(dit, file)')
    file.close()
    # Dot
    file = open(f'{path}/Ex_Two_Time_{seed}_{trajectories}_{steps}_{mule}', 'wb')
    exec(f'pickle.dump(dot, file)')
    file.close()

#%%# BiochemAnalysisMule

seed = 25
trajectories = 1000
steps = 10000
mule = 50
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Data'

file = open(f'{path}/Ex_One_{seed}_{trajectories}_{steps}_{mule}', 'rb')
exec(f'dit = pickle.load(file)')
file.close()

file = open(f'{path}/Ex_One_Time_{seed}_{trajectories}_{steps}_{mule}', 'rb')
exec(f'dot = pickle.load(file)')
file.close()

dark = ('darkblue', 'darkorange', 'darkgreen', 'darkred', 'violet', 'maroon', 'hotpink', 'darkgray', 'darkolivegreen', 'darkturquoise')

# Dit

coves = list(dit.keys())
mean_NA = [dit.get(_)[0] for _ in dit.keys()]
stan_err_NA = [dit.get(_)[1] for _ in dit.keys()]
x = coves
y = mean_NA
z = [np.abs((np.array(y)+np.array(stan_err_NA))-0.5).argmin(), np.abs(np.array(y)-0.5).argmin(), np.abs((np.array(y)-np.array(stan_err_NA))-0.5).argmin()]

plt.errorbar(x, y, yerr = stan_err_NA, uplims = True, lolims = True, color = dark[1])
plt.xlabel('N\nCopy Number')
plt.ylabel('NA\nRelative Time ON')
plt.ylim(-np.max(stan_err_NA), 1+np.max(stan_err_NA))
plt.title(f'Relative Time ON\nTrajectories: {trajectories}        Steps: {steps}')
plt.hlines(y[z[1]], 0, max(x), color = 'lightblue')
plt.vlines([x[z[_]] for _ in range(len(z))], -np.max(stan_err_NA), 1+np.max(stan_err_NA), color = 'lightblue')
print('x:', [x[z[_]] for _ in range(len(z))], '\t', 'y:', y[z[1]])

hill_fun = _hill_fun(x[z[1]]) # Curve fitting!
opt = optimize.curve_fit(hill_fun, x, y, coop)
print('Hill Coefficient:', opt[0])
plt.plot(x, hill_fun(x, opt[0]), color = dark[0])
ran = np.random.default_rng(seed = seed) # Alternative approach!
y_res = [ran.normal(loc, scale) for loc, scale in dit.values()]
opt_res = optimize.curve_fit(hill_fun, x, y_res, coop)
print('Hill Coefficient:', opt_res[0], '\tAlternative approach!')
plt.plot(x, hill_fun(x, opt_res[0]), color = dark[3])

plt.show()

# Dot

coves = list(dot.keys())
mean_time = [dot.get(_)[0] for _ in dot.keys()]
stan_deva_time = [dot.get(_)[1] for _ in dot.keys()]
x = coves
y = mean_time
z = [np.nanargmin(np.array(y)+np.array(stan_deva_time)), np.nanargmin(y), np.nanargmin(np.array(y)-np.array(stan_deva_time))]

plt.errorbar(x, y, yerr = stan_deva_time, uplims = True, lolims = True, color = dark[2])
plt.xlabel('N\nCopy Number')
plt.ylabel('Simulation Time\nHours')
plt.ylim(-np.nanmax(stan_deva_time), np.nanmax(y)+np.nanmax(stan_deva_time))
plt.title(f'Simulation Time\nTrajectories: {trajectories}        Steps: {steps}')
plt.hlines(y[z[1]], 0, max(x), color = 'hotpink')
plt.vlines([x[z[_]] for _ in range(len(z))], -np.nanmax(stan_deva_time), np.nanmax(y)+np.nanmax(stan_deva_time), color = 'hotpink')
print('x:', [x[z[_]] for _ in range(len(z))], '\t', 'y:', y[z[1]])
