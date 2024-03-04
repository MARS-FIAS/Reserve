###############################################
########                               ########
######## NANOG / GATA [Extra Analysis] ########
########                               ########
###############################################

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
# hack = 250 # Half Activation
hare = 1000 # Half Repression
print('\n\n\tQ = ' + str(q) + '\t|\tHalf-Repression Threshold = ' + str(int(hare/10)) + '\n\n')
kb_promoter = {art: 250*kf_promoter if 'A' in art else hare*kf_promoter*q if 'N' in art else hare*kf_promoter/q for art in arts}

_protein_lifetime = (5, 6, 7) # Hours
protein_lifetime = _protein_lifetime[1]*pow(60, 2) # Seconds
protein_copy_number = 100 # Steady-state (old) assumption # kf/kb = 1000

share = 0.8 # (0, 1)
kf_protein = share*protein_copy_number/protein_lifetime
kb_protein = 1/protein_lifetime
kf_spontaneous = (1-share)*protein_copy_number/protein_lifetime # kf_protein/10 # 10*kb_protein # mRNA leakage
kb_spontaneous = kb_protein

#%%# Biochemical System Construction

# Spontaneous Reactions

# Forward

soft_exes = [f'0 -> {P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
soft_props = [f'(1-np.sign({P}I))*kf_{P}0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
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
trajectories = 1000
steps = 250000
press = BiochemSimulMule(proto, steps, trajectories, seed)
%time press.meth_direct()
# Individual # Collective

#%%# Anime [Demo!]

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
h = pow(60, 2)
d = 24
k = 7

a = np.max(press.epoch_mat, axis = 0)/h/d
plt.plot(a)
plt.show()
s = [0, 1] # species = ['N', 'G']
j = 50
t = press.epoch_mat[:, j]
z = press.state_tor[:, s, j]
plt.plot(t[t <= k*d*h]/h/d, z[t <= k*d*h, :])
plt.show()

#%%# Anime Data

self = press # Simulation
species = ['N', 'G']
s = [0, 1]
show = False
teds = [pow(60, 2), 24, 5] # teds = [pow(60, 2), 24, 7]
tie = np.linspace(0, int(teds[0]*teds[1]*teds[2]), int(teds[0]*teds[1]*teds[2])+1)
stamp = 1
ties = np.array([stamp*h for h in range(1, int(teds[1]/stamp)*teds[2]+stamp) if stamp*h <= teds[1]*teds[2]]) # Hours
sties = teds[0]*ties # Seconds
trajectories = range(self.state_tor.shape[2]) if not show else range(10)
conus = np.full((len(species), len(ties), len(trajectories)), np.nan)
maxi = np.max(self.state_tor[:, s, :])

for trajectory in trajectories:
    x = self.epoch_mat[:, trajectory]
    y = self.state_tor[:, s, trajectory]
    fun = interpolate.interp1d(x = x, y = y, kind = 0, axis = 0)
    z = fun(tie)
    conus[:, :, trajectory] = np.transpose(z[sties])
    if show:
        plt.plot(tie/teds[0]/teds[1], z)
        plt.plot(tie[sties]/teds[0]/teds[1], z[sties])
        plt.show()

#%%# Anime Plot [Three]

safe = False
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Data/Share'
sepal = 25
cum = False

cocos = np.linspace(0, 1, len(trajectories))
cam = matplotlib.cm.get_cmap('Spectral') # ['Spectral', 'RdYlGn', 'PiYG', 'coolwarm']

cos = ['tab:red', 'tab:olive', 'tab:cyan', 'tab:purple']
mares = ['.', '<', '>', '*']
mu = np.mean(conus, 2)
sigma = np.std(conus, 2)

tempi = [np.unique(conus[0], return_counts = True)[1], np.unique(conus[1], return_counts = True)[1]]
tops = [2*np.max(temp/np.sum(temp)) for temp in tempi]
top = np.round(np.max(tops), 2) if not cum else 1.05

beg = 0
end = protein_copy_number
clue = np.repeat(1/(end-beg), end-beg)
bend = np.arange(beg, end)

for t in range(len(ties)):
    tit = f'Auto = {share} + Basal = {np.round(1-share, 2)} @ Hart = {int(hare/10)} @ Time | E{np.round(ties[t]/teds[1], 1)} = {ties[t]} Hours'
    # tit = f'Autoactivation = {share} + Basal = {np.round(1-share, 2)} @ Q = {q} @ Time @ E{np.round(ties[t]/teds[1], 1)} = {ties[t]} Hours'
    fig, (axe, axi) = plt.subplots(2, 2, constrained_layout = True)
    fig.suptitle(tit)
    # (0, 0) # Individual
    axe[0].set_aspect('equal')
    axe[0].set_xlim(0, maxi)
    axe[0].set_ylim(0, maxi)
    axe[0].set_xlabel(species[0])
    axe[0].set_ylabel(species[1])
    x = conus[0, t, :]
    y = conus[1, t, :]
    axe[0].scatter(x, y, c = cocos, cmap = cam)
    axe[0].axhline(sepal, 0, maxi, color = 'gray')
    axe[0].axvline(sepal, 0, maxi, color = 'gray')
    axe[0].axline((0, 0), slope = 1, color = 'lightgray')
    # (0, 1) # Collective
    axe[1].set_aspect('equal')
    axe[1].set_xlim(0, maxi)
    axe[1].set_ylim(0, maxi)
    axe[1].set_xlabel(species[0])
    axe[1].set_ylabel(species[1])
    x = mu[0, t]
    y = mu[1, t]
    axe[1].scatter(x, y, s = 200, color = cos[0], marker = mares[0])
    x = mu[0, t] + sigma[0, t]
    y = mu[1, t] + sigma[1, t]
    axe[1].scatter(x, y, s = 100, color = cos[1], marker = mares[1])
    x = np.max([0, mu[0, t] - sigma[0, t]])
    y = np.max([0, mu[1, t] - sigma[1, t]])
    axe[1].scatter(x, y, s = 100, color = cos[2], marker = mares[2])
    axe[1].axhline(sepal, 0, maxi, color = 'gray')
    axe[1].axvline(sepal, 0, maxi, color = 'gray')
    axe[1].axline((0, 0), slope = 1, color = 'lightgray')
    axe[1].grid(True, color = 'lavender', linestyle = 'dashed')
    # (1, 0) # Collective
    axi[0].set_xlim(-1, np.max(conus[0]))
    axi[0].set_ylim(0, top)
    axi[0].set_xlabel(species[0])
    axi[0].set_ylabel('Relative Frequency')
    x = conus[0, t, :]
    axi[0].hist(x, int(np.max(x)), density = True, cumulative = cum, align = 'mid', color = 'tab:blue')
    axi[0].axvline(sepal, 0, maxi, color = 'darkgray')
    if cum:
        axi[0].plot(bend, np.cumsum(clue), color = 'lightgray', linestyle = '--')
        axi[0].vlines(end-1, 0, 1, color = 'lightgray', linestyle = '--')
    else:
        axi[0].plot(bend, clue, color = 'lightgray', linestyle = '--')
        axi[0].vlines([beg, end-1], 0, 1/(end-beg), color = 'lightgray', linestyle = '--')
    # (1, 1) # Collective
    axi[1].set_xlim(-1, np.max(conus[1]))
    axi[1].set_ylim(0, top)
    axi[1].set_xlabel(species[1])
    axi[1].set_ylabel('Relative Frequency')
    y = conus[1, t, :]
    axi[1].hist(y, int(np.max(y)), density = True, cumulative = cum, align = 'mid', color = 'tab:orange')
    axi[1].axvline(sepal, 0, maxi, color = 'darkgray')
    if cum:
        axi[1].plot(bend, np.cumsum(clue), color = 'lightgray', linestyle = '--')
        axi[1].vlines(end-1, 0, 1, color = 'lightgray', linestyle = '--')
    else:
        axi[1].plot(bend, clue, color = 'lightgray', linestyle = '--')
        axi[1].vlines([beg, end-1], 0, 1/(end-beg), color = 'lightgray', linestyle = '--')
    if safe:
        plt.savefig(path+'/'+tit+' @ '+str(cum)+'.jpeg', dpi = 250)
    plt.show()

#%%# Promoter Switching [Demo!]

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
h = pow(60, 2)
d = 24
k = 7

a = np.max(press.epoch_mat, axis = 0)/h/d
plt.plot(a)
plt.show()
s = [22, 24] # species = ['NA', 'GA']
j = 50
t = press.epoch_mat[:, j]
z = press.state_tor[:, s, j]
plt.plot(t[t <= k*d*h]/h/d, z[t <= k*d*h, :])
plt.show()

#%%# Promoter Switching Data

self = press # Simulation
species = ['NA', 'GA']
s = [22, 24]
show = True
teds = [1, 24*pow(60, 2), 5] # teds = ['seconds', {'hours', 'minutes', 'seconds'}, 'days']
tie = np.linspace(0, int(teds[0]*teds[1]*teds[2]), int(teds[0]*teds[1]*teds[2])+1) # Seconds
stamp = 900 # {'seconds', 'minutes', 'hours', 'days'}
ties = np.array([stamp*h for h in range(1, int(teds[1]/stamp)*teds[2]+stamp) if stamp*h <= teds[1]*teds[2]]) # {'seconds', 'minutes', 'hours', 'days'}
sties = teds[0]*ties # Seconds
trajectories = range(self.state_tor.shape[2]) if not show else range(10)
conus = np.full((len(species), len(ties), len(trajectories)), np.nan)
maxi = np.max(self.state_tor[:, s, :])

for trajectory in trajectories:
    x = self.epoch_mat[:, trajectory]
    y = self.state_tor[:, s, trajectory]
    fun = interpolate.interp1d(x = x, y = y, kind = 0, axis = 0)
    z = fun(tie)
    conus[:, :, trajectory] = np.transpose(z[sties])
    if show:
        plt.plot(tie/teds[0]/teds[1], z)
        plt.plot(tie[sties]/teds[0]/teds[1], z[sties])
        plt.show()
