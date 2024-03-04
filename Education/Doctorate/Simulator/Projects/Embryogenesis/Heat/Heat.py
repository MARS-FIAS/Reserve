#%%# Configuration

# BiochemStem/Simul/Analysis (Support) # BiochemStemFun_MRNA_Heat (Support) # Embryogenesis_Shutdown_MRNA_Heat (Inspiration)

#%%# Catalyze

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

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Auxiliaries

def instate_fun(seed, cycle, start, keep = False, bid = False, pro = 0.5, special = False):
    if cycle <= start:
        ret = None
    else:
        where = [list(press.stem.assembly['species'].values()).index(_) for _ in ['N_MRNA', 'G_MRNA', 'N', 'G']]
        _nowhere = {_ for _ in range(press.state_tor[press.mate_index, :, :].shape[0])}
        nowhere = list(_nowhere.difference(where))
        if keep:
            alp = press.state_tor[press.mate_index, :, :].copy()
            bet = alp.copy()
            if bid:
                print('Bid!'+' | '+str(pro))
                bet[where, :] = np.random.default_rng(seed = seed).binomial(n = bet[where, :], p = pro, size = (len(where), bet.shape[1])).astype('uint16')
            else:
                print('Equal!')
                bet[where, :] = np.floor(alp[where, :]*pro).astype('uint16')
            alp[where, :] = alp[where, :] - bet[where, :]
            ret = bet if special else np.concatenate((alp, bet), 1)
        else:
            original = np.array(list(press.stem.initial_state.values())).reshape((-1, 1))
            alp = press.state_tor[press.mate_index, :, :].copy()
            alp[nowhere, :] = np.copy(original[nowhere, :])
            bet = alp.copy()
            if bid:
                print('Bid!'+' | '+str(pro))
                bet[where, :] = np.random.default_rng(seed = seed).binomial(n = bet[where, :], p = pro, size = (len(where), bet.shape[1])).astype('uint16')
            else:
                print('Equal!')
                bet[where, :] = np.floor(alp[where, :]*pro).astype('uint16')
            alp[where, :] = alp[where, :] - bet[where, :]
            ret = bet if special else np.concatenate((alp, bet), 1)
    return ret

def sap_prop_fun(conus, t, testa, sepal):
    N_N = conus[0, t, testa] < sepal
    N_P = conus[0, t, testa] >= sepal
    G_N = conus[1, t, testa] < sepal
    G_P = conus[1, t, testa] >= sepal
    NG_NN = np.count_nonzero(N_N * G_N)
    NG_PN = np.count_nonzero(N_P * G_N)
    NG_NP = np.count_nonzero(N_N * G_P)
    NG_PP = np.count_nonzero(N_P * G_P)
    ret = [NG_NN, NG_PN, NG_NP, NG_PP]
    return ret

dis = range(5) # range(10)
pro = 0.4 # p = 0.3 # p = 10/25
volumes = [200000/np.power(2, d) for d in dis] # Microns
volumes[4] = volumes[3]*pro
volumes[5:] = [volumes[4]/2]*3
radii = [np.power(3*volume/(4*np.pi), 1/3) for volume in volumes]
cycle_steps = [50000, 50000, 50000, 50000, 50000, 100000, 100000, 100000, 250000, 100000]
mates = [24, 24, 12, 12, 12, 12, 12, 12, 24, 24]
_shutdowns = {cycle: [(0, 12), (22, 24)] if cycle == 0 else [(0, 2), (10, 12)] for cycle in dis} # Hours!
plt.plot(dis, volumes, color = 'tab:blue')
plt.show()
plt.plot(dis, radii, color = 'tab:orange')
plt.hlines(5, 0, len(dis), color = 'tab:gray')

#%%# Cell Division / Cell Volume Change

# cam = matplotlib.cm.get_cmap('Spectral') # ['Spectral', 'RdYlGn', 'PiYG', 'coolwarm']
epoch = 0
vara = [0 for _ in dis]
reps = 100 # How many embryos do you wish?
asymmetric = 4 # Asymmetrical division! Which cycle?
special = False

_scope_alp = np.round(np.linspace(1.25, 2.25, 11), 2) # Repress # [0.25, 1.25]
_scope_bet = np.round(np.linspace(0.25, 2.25, 21), 2) # Activate # [1.25, 2.25]
scope = [(alp, bet) for alp in _scope_alp for bet in _scope_bet]

simas = reps # len(dis) = cycles
data = {(key, key_0, key_1): None for key in scope for key_0 in range(simas) for key_1 in dis}

for mules in scope:
    print(f'\n\n{mules}\n\n')
    for cycle in dis: # dis:
        
        if cycle == asymmetric:
            pro = 0.4 # p in {0.3, 0.4} # p = 10/25
            special = True
        else:
            pro = 0.5
            special = False
        
        seed = 27*(scope.index(mules)+1) + cycle
        vomer = True if cycle in [5, 6, 7] else False # Activate the inter-cycle volume change!
        print(vomer)
        mate = mates[cycle]*pow(60, 2) # Hours
        trajectories = np.power(2, cycle-1)*reps if cycle >= asymmetric else np.power(2, cycle)*reps
        print(trajectories)
        steps = cycle_steps[cycle] # 50000
        start = 0 # Starting cycle! It will simply ignore the previous cycles!
        keep = False # Keep the state of the promoter?
        bid = True # Binomial Distribution?
        instate = instate_fun(seed, cycle, start, keep, bid, pro, special)
        
        ################
        
        radius = radii[cycle]
        _pol = (2, 4, 6, 8)
        pol = 1 # {0, 1, 2, 3, 4}
        _MRNA = 250
        _protein = 4
        # mules = (1, 1)
        shutdowns = [] # [(alp*pow(60, 2), bet*pow(60, 2)) for alp, bet in _shutdowns[cycle]]
        proto = BiochemStemFun(radius, pol, _MRNA, _protein, mules)
        vara[cycle] = proto.rates.copy() # Vara Test!
        press = BiochemSimulMuse(proto, instate, steps, trajectories, mate, vomer, shutdowns, seed)
        press.meth_direct()
        
        ################
        
        self = press # Simulation
        species = ['N', 'G']
        s = [28, 29]
        show = False
        teds = [pow(60, 2), 24, mates[cycle]/24] # teds = [pow(60, 2), 24, 7]
        tie = np.linspace(0, int(teds[0]*teds[1]*teds[2]), int(teds[0]*teds[1]*teds[2])+1)
        stamp = 1
        ties = np.array([stamp*h for h in range(0, int((teds[1]/stamp)*teds[2])+stamp) if stamp*h <= teds[1]*teds[2]]) # Hours
        sties = teds[0]*ties # Seconds
        trajectories = range(self.state_tor.shape[2]) if not show else range(np.power(2, cycle))
        conus = np.full((len(species), len(ties), len(trajectories)), np.nan)
        maxi = 10*125 # np.max(self.state_tor[:, s, :])
        
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
        
        ######## Collect population proportion estimation!
        
        simas = np.arange(0, len(trajectories), 1, int) # simas = np.arange(0, len(trajectories)+1, len(trajectories)/reps, int)
        for sima in range(reps): # range(len(simas)-1)
            here = np.argwhere(simas % reps == sima).reshape(-1).tolist()
            sample_proportions = np.zeros((len(ties), 4))
            for t in range(len(ties)):
                sepal = 250
                sample_proportions[t] = sap_prop_fun(conus[..., here], t, ..., sepal)
            data[(mules, sima, cycle)] = sample_proportions
        
        ######## Collect population proportion estimation!
        
        epoch = epoch + mates[cycle]

#%%# Store Data

safe = False
path = '/home/mars-fias/Downloads/Clue_Data'

if safe:
    file = open(f'{path}/{_MRNA}_{len(scope)}_{reps}_conus', 'wb')
    exec(f'pickle.dump(data, file)')

#%%# Load Data [Info]

zea = 250
alp = 121
bet = 100
file = open(f'{path}/{zea}_{alp}_{bet}_conus', 'rb')
exec(f'info = pickle.load(file)')
file.close()

#%%# Plot Info Intro

scope = list({_[0] for _ in info.keys()})
scope.sort()
simas = len({_[1] for _ in info.keys()})
cycles = list({_[2] for _ in info.keys()})

_info = {(key, key_0, key_1): None for key in scope for key_0 in range(simas) for key_1 in cycles}
stars = np.full((len(scope), len(cycles)), np.nan, 'O')

def loss(w):
    ex = 2
    t = np.sum(w)
    n = w[1]
    g = w[2]
    x = np.power(n+g, ex)/np.power(t, ex)
    y = np.power(np.abs(n-g), ex)/np.power(t, ex)
    z = x-y
    return z

# Calculate my loss!
for mules in scope:
    for sima in range(simas):
        for cycle in cycles:
            temp = info[(mules, sima, cycle)]
            _info[(mules, sima, cycle)] = np.apply_along_axis(loss, 1, temp)

# Use my loss to create a heat map!
for mules in scope:
    for cycle in cycles:
        for sima in range(simas):
            temp = _info[(mules, sima, cycle)]
            if sima == 0:
                extra = temp
            else:
                extra = np.vstack((extra, temp))
        stars[scope.index(mules), cycle] = extra.mean(0) # Adapter!

#%%# Plot Info

import seaborn as sns

_scope_alp = np.sort(np.array(list({_[0] for _ in scope})))
_scope_bet = np.sort(np.array(list({_[1] for _ in scope})))
Y = _scope_alp
X = _scope_bet

plt.rcParams['figure.figsize'] = (7, 7) # (5, 5)
# link = https://matplotlib.org/stable/tutorials/colors/colormaps.html
cap = 1

safe = False
testa = '(cycle+1 == 4 and t == 12) or (cycle+1 == 5)'
epoch = 0
for cycle in cycles:
    for index in range(len(scope)):
        print(scope[index], cycle)
        if index == 0:
            temp = stars[index, cycle]
        else:
            temp = np.vstack((temp, stars[index, cycle]))
    for t in range(temp.shape[1]):
        Z = temp[:, t].reshape((len(Y), len(X))) # vmax = reps
        axe = sns.heatmap(data = Z, vmin = 0, vmax = 1, cmap = 'hot', annot = False, square = True, linewidth = 0, xticklabels = cap*X, yticklabels = cap*Y)
        axe.invert_yaxis()
        tit = f'Cycle = {cycle+1}\n# Cells = {np.power(2, cycle)}|{np.power(2, cycle-1) if cycle >= asymmetric else np.power(2, cycle)} | Inter-Division Time = {mates[cycle]} Hours\nTime = [{np.round(epoch/24, 1)}, {np.round((epoch + mates[cycle])/24, 1)}] Days\nHour = {t}'
        plt.title(tit)
        plt.xlabel('Activate')
        plt.ylabel('Repress')
        if safe and eval(testa):
            plt.savefig(f'{path}/Heat_{cycle+1}_{t}_Mean_True.jpeg')
        plt.show()
    epoch = epoch + mates[cycle]

#%%# Plot Info Extra

import seaborn as sns

_scope_alp = np.sort(np.array(list({_[0] for _ in scope})))
_scope_bet = np.sort(np.array(list({_[1] for _ in scope})))
Y = _scope_alp
X = _scope_bet

plt.rcParams['figure.figsize'] = (8, 7) # (5, 4)
# link = https://matplotlib.org/stable/tutorials/colors/colormaps.html
coloring = list(matplotlib.colors.TABLEAU_COLORS)
cap = 1

safe = False
testa = '(cycle+1 == 4 and t == 12) or (cycle+1 == 5)'
epoch = 0
for cycle in cycles:
    for index in range(len(scope)):
        print(scope[index], cycle)
        if index == 0:
            temp = stars[index, cycle]
        else:
            temp = np.vstack((temp, stars[index, cycle]))
    for t in range(temp.shape[1]):
        Z = temp[:, t].reshape((len(Y), len(X)))
        #
        fig, axe = plt.subplots(1, 1, constrained_layout = True)
        tit = f'Cycle = {cycle+1}\n# Cells = {np.power(2, cycle)}|{np.power(2, cycle-1) if cycle >= asymmetric else np.power(2, cycle)} | Inter-Division Time = {mates[cycle]} Hours\nTime = [{np.round(epoch/24, 1)}, {np.round((epoch + mates[cycle])/24, 1)}] Days\nHour = {t}'
        fig.suptitle(tit) # vmax = reps # levels = np.linspace(0, reps, int(2*reps/10+1))
        cor = axe.contourf(X, Y, Z, cmap = 'hot', vmin = 0, vmax = 1, levels = np.linspace(0, 1, int(2*reps/10+1)))
        # cor = axe.contourf(X, Y, Z, cmap = 'hot', vmin = 0, vmax = reps)
        car = fig.colorbar(mappable = cor, ax = axe, location = 'right')
        # car.ax.set_ylim(0, reps)
        axe.grid(color = coloring[7], linestyle = '--', alpha = 0.1)
        axe.set_xlabel('Activate')
        axe.set_xticks(X)
        axe.set_ylabel('Repress')
        axe.set_yticks(Y)
        if safe and eval(testa):
            plt.savefig(f'{path}/Cone_{cycle+1}_{t}_Mean.jpeg')
        plt.show()
        #
    epoch = epoch + mates[cycle]
