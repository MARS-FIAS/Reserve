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

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# BiochemSimul [Test One]

seed = 25 # 73
vomer = True
mate = 1*24*pow(60, 2) # Hours
trajectories = 1
steps = 10
instate = None
shutdowns = [(0, 0.5), (7.5, 8)]
proto = BiochemStemFun(7, 0)
press = BiochemSimulMuse(proto, instate, steps, trajectories, mate, vomer, shutdowns, seed)
press.meth_direct()

z = press.epoch_mat[:, 0]
plt.plot(np.nanmax(z)*np.diff(z)/np.nanmax(np.diff(z)))
plt.plot(z) # Careful!
plt.show()

h = pow(60, 2)
d = 24
k = 5
s = [0, 1] # species = ['N', 'G']
j = 0
t = press.epoch_mat[:, j]
z = press.state_tor[:, s, j]
plt.plot(t[t <= k*d*h]/h/d, z[t <= k*d*h, :])
plt.show()

#%%# BiochemSimul [Test Two]

seed = 25 # 73
vomer = False
mate = 24*pow(60, 2) # Hours
trajectories = 1
steps = 10

instate = press.state_tor[press.mate_index, :, :].copy()

press = BiochemSimulMuse(proto, instate, steps, trajectories, mate, vomer, shutdowns, seed)
press.meth_direct()

z = press.epoch_mat[:, 0]
plt.plot(np.nanmax(z)*np.diff(z)/np.nanmax(np.diff(z)))
plt.plot(z) # Careful!
plt.show()
plt.plot(press.state_tor[:, 0, 0])
plt.show()

#%%# Color Test

trajectories = pow(2, 10)
temp = matplotlib.cm
_cam = np.linspace(0, 1, trajectories)
cam = temp.jet(_cam)
plt.hlines(y = list(range(trajectories)), xmin = 0, xmax = 1, colors = cam)

#%%# Cell Volumes

dis = range(8) # range(10)

pro = 0.4 # p = 0.3 # p = 10/25
volumes = [200000/np.power(2, d) for d in dis] # Microns
volumes[4] = volumes[3]*pro
volumes[5:] = [volumes[4]/2]*3
radii = [np.power(3*volume/(4*np.pi), 1/3) for volume in volumes]
cycle_steps = [3000, 4000, 5000, 5000, 5000, 10000, 25000, 50000, 75000, 100000]
mates = [24, 24, 12, 12, 12, 12, 12, 12, 24, 24]

_shutdowns = {cycle: [(0, 12), (22, 24)] if cycle == 0 else [(0, 2), (10, 12)] for cycle in dis} # Hours!
# _shutdowns = {cycle: [(0, 0.5), (23.5, 24)] if cycle in [0, 1] else [(0, 0.5), (11.5, 12)] for cycle in dis} # Hours!

plt.plot(dis, volumes, color = 'tab:blue')
plt.show()
plt.plot(dis, radii, color = 'tab:orange')
plt.hlines(5, 0, len(dis), color = 'tab:gray')

#%%# Instate Fun

def instate_fun(seed, cycle, start, keep = False, bid = False, pro = 0.5, special = False):
    
    if cycle <= start:
        ret = None
    else:
        if keep:
            alp = press.state_tor[press.mate_index, :, :].copy()
            bet = alp.copy()
            if bid:
                print('Bid!'+' | '+str(pro))
                bet[:2, :] = np.random.default_rng(seed = seed).binomial(n = bet[:2, :], p = pro, size = (2, bet.shape[1])).astype('uint16')
            else:
                print('Equal!')
                bet[:2, :] = np.floor(alp[:2, :]*pro).astype('uint16')
            alp[:2, :] = alp[:2, :] - bet[:2, :]
            ret = bet if special else np.concatenate((alp, bet), 1)
        else:
            original = np.array(list(press.stem.initial_state.values())).reshape((-1, 1))
            alp = press.state_tor[press.mate_index, :, :].copy()
            alp[2:, :] = np.copy(original[2:, :])
            bet = alp.copy()
            if bid:
                print('Bid!'+' | '+str(pro))
                bet[:2, :] = np.random.default_rng(seed = seed).binomial(n = bet[:2, :], p = pro, size = (2, bet.shape[1])).astype('uint16')
            else:
                print('Equal!')
                bet[:2, :] = np.floor(alp[:2, :]*pro).astype('uint16')
            alp[:2, :] = alp[:2, :] - bet[:2, :]
            ret = bet if special else np.concatenate((alp, bet), 1)
    return ret

#%%# Sap Prop Fun [Anime Plot Fun!]

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

#%%# Strap Fun [Anime Plot Fun!]

# seed = 73
# t = 12
# sepal = 25
# replicas = 10000
# cos = ['tab:red', 'tab:olive', 'tab:cyan', 'tab:purple']

def strap_fun(seed, t, sepal, replicas, cos):
    
    bootstrap_replications = np.zeros(shape = (replicas, 4))
    sample_proportion = sap_prop_fun(conus, t, ..., sepal)

    for replica in range(replicas):
        # testa = bootstrap_samples
        testa = np.random.default_rng(seed = seed+replica).integers(low = 0, high = conus.shape[2], size = conus.shape[2])
        ret = sap_prop_fun(conus, t, testa, sepal)
        bootstrap_replications[replica, :] = ret
    
    confidence = 0.95
    
    delta = sample_proportion - bootstrap_replications
    _l = np.percentile(a = delta, q = 100*(1-confidence)/2, axis = 0, interpolation = 'nearest')
    _u = np.percentile(a = delta, q = 100*(1+confidence)/2, axis = 0, interpolation = 'nearest')
    l = np.abs(_l/len(trajectories))
    u = _u/len(trajectories)
    
    return (sample_proportion, l, u)

# sample_proportion, l, u = strap_fun(replicas)

# x = list(range(1, 5))
# height = np.array(sample_proportion)/len(trajectories)
# tick_label = ['N-G-', 'N+G-', 'N-G+', 'N+G+']
# #
# #plt.set_aspect('auto')
# plt.xlim(0, 5)
# plt.ylim(0, 1)
# plt.xlabel('Class')
# plt.ylabel('Proportion')
# plt.grid(axis = 'y', which = 'major', linestyle = '--')
# plt.bar(x = x, height = height, yerr = [l, u], tick_label = tick_label, color = cos)
# plt.show()
# print(sample_proportion)

#%%# Anime Plot Fun!

def anime_fun(opt):
    
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
    end = 100
    clue = np.repeat(1/(end-beg), end-beg)
    bend = np.arange(beg, end)
    
    for t in range(len(ties)):
        tit = f'Cycle = {cycle+1} | Binomial = {bid}\n# Cells = {np.power(2, cycle)}|{np.power(2, cycle-1) if cycle >= asymmetric else np.power(2, cycle)} | Inter-Division Time = {mates[cycle]} Hours\nTime = [{np.round(epoch/24, 1)}, {np.round((epoch + mates[cycle])/24, 1)}] Days | Lifetime = {_pol[pol]} Hours\n{t}'
        fig, (axe, axi) = plt.subplots(1, 2, constrained_layout = True)
        fig.suptitle(tit)
        # (0, 0) # Individual
        axe.set_aspect('equal')
        axe.set_xlim(0, maxi)
        axe.set_ylim(0, maxi)
        axe.set_xlabel(species[0])
        axe.set_ylabel(species[1])
        x = conus[0, t, :]
        y = conus[1, t, :]
        axe.scatter(x, y, c = cocos, cmap = cam)
        axe.axhline(sepal, 0, maxi, color = 'gray')
        axe.axvline(sepal, 0, maxi, color = 'gray')
        axe.axline((0, 0), slope = 1, color = 'lightgray')
        # (0, 1) # Collective
        if opt == 'scat':
            axi.set_aspect('equal')
            axi.set_xlim(0, maxi)
            axi.set_ylim(0, maxi)
            axi.set_xlabel(species[0])
            axi.set_ylabel(species[1])
            x = mu[0, t]
            y = mu[1, t]
            axi.scatter(x, y, s = 200, color = cos[0], marker = mares[0])
            x = mu[0, t] + sigma[0, t]
            y = mu[1, t] + sigma[1, t]
            axi.scatter(x, y, s = 100, color = cos[1], marker = mares[1])
            x = np.max([0, mu[0, t] - sigma[0, t]])
            y = np.max([0, mu[1, t] - sigma[1, t]])
            axi.scatter(x, y, s = 100, color = cos[2], marker = mares[2])
            axi.axhline(sepal, 0, maxi, color = 'gray')
            axi.axvline(sepal, 0, maxi, color = 'gray')
            axi.axline((0, 0), slope = 1, color = 'lightgray')
            axi.grid(True, color = 'lavender', linestyle = 'dashed')
        elif opt == 'bar':
            #
            replicas = 1*len(trajectories)
            print(replicas)
            sample_proportion, l, u = strap_fun(seed, t, sepal, replicas, cos)
            x = list(range(1, 5))
            height = np.array(sample_proportion)/len(trajectories)
            tick_label = ['N-G-', 'N+G-', 'N-G+', 'N+G+']
            #
            axi.set_aspect('auto')
            axi.set_xlim(0, 5)
            axi.set_ylim(0, 1)
            axi.set_xlabel('Class')
            axi.set_ylabel('Proportion')
            axi.bar(x = x, height = height, yerr = [l, u], tick_label = tick_label, color = cos)
            axi.grid(b = True, axis = 'y', color = 'lavender', linestyle = 'dashed')
        if safe:
            plt.savefig(path+'/'+tit+' @ '+str(cum)+'.jpeg', dpi = 250)
        plt.show()
    
    return None

#%%# Cell Division / Cell Volume Change

# cam = matplotlib.cm.get_cmap('Spectral') # ['Spectral', 'RdYlGn', 'PiYG', 'coolwarm']
epoch = 0
vara = [0 for _ in dis]
reps = 1 # How many embryos do you wish?
asymmetric = 4 # Asymmetrical division! Which cycle?
special = False

simas = 100 # len(dis) = cycles
data = {(key_0, key_1): None for key_0 in range(simas) for key_1 in dis}

for sima in range(simas):
    print(f'\n{sima}\n')
    for cycle in dis: # dis:
        
        if cycle == asymmetric:
            pro = 0.4 # p in {0.3, 0.4} # p = 10/25
            special = True
        else:
            pro = 0.5
            special = False
        
        seed = 227*(sima+1) + cycle
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
        shutdowns = [] # [(alp*pow(60, 2), bet*pow(60, 2)) for alp, bet in _shutdowns[cycle]]
        proto = BiochemStemFun(radius, pol)
        vara[cycle] = proto.rates.copy() # Vara Test!
        press = BiochemSimulMuse(proto, instate, steps, trajectories, mate, vomer, shutdowns, seed)
        press.meth_direct()
        
        # z = press.epoch_mat[:, 0]
        # plt.plot(np.nanmax(z)*np.diff(z)/np.nanmax(np.diff(z)))
        # plt.plot(z) # Careful!
        # plt.show()
        
        ################
        
        self = press # Simulation
        species = ['N', 'G']
        s = [0, 1]
        show = False
        teds = [pow(60, 2), 24, mates[cycle]/24] # teds = [pow(60, 2), 24, 7]
        tie = np.linspace(0, int(teds[0]*teds[1]*teds[2]), int(teds[0]*teds[1]*teds[2])+1)
        stamp = 1
        ties = np.array([stamp*h for h in range(0, int((teds[1]/stamp)*teds[2])+stamp) if stamp*h <= teds[1]*teds[2]]) # Hours
        sties = teds[0]*ties # Seconds
        trajectories = range(self.state_tor.shape[2]) if not show else range(np.power(2, cycle))
        conus = np.full((len(species), len(ties), len(trajectories)), np.nan)
        maxi = 125 # np.max(self.state_tor[:, s, :])
        
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
        
        opts = ['scat', 'bar']
        opt = opts[1]
        # anime_fun(opt)
        
        ######## Collect population proportion estimation!
        
        sample_proportions = np.zeros((len(ties), 4))
        for t in range(len(ties)):
            sepal = 25
            sample_proportions[t] = sap_prop_fun(conus, t, ..., sepal)
        data[(sima, cycle)] = sample_proportions
        
        ######## Collect population proportion estimation!
        
        ################ Single Trajectories!
        # h = pow(60, 2)
        # d = 24
        
        # _cam = np.linspace(0, 1, trajectories)
        # cam = matplotlib.cm.jet(_cam)
        # tit = f'# Cells = {np.power(2, cycle)} | Inter-Division Time = {mates[cycle]} Hours\nTime = [{np.round(epoch/24, 1)}, {np.round((epoch + mates[cycle])/24, 1)}] Days'
        
        # s = [1] # species = ['N', 'G']
        # js = range(np.power(2, cycle)*reps)
        # for j in js:
        #     t = press.epoch_mat[:, j]
        #     z = press.state_tor[:, s, j]
        #     plt.plot(t[t <= mate]/h/d, z[t <= mate, :]) # color = cam[j, :]
        #     plt.ylim(-1, 121)
        # plt.title(tit)
        # plt.show()
        # # testa = press.state_tor[:, s, :]
        # # at = 1 if cycle == 0 else 2
        # # plt.plot(t[t <= mate]/h/d, np.amin(testa[t <= mate], at))
        # # plt.plot(t[t <= mate]/h/d, np.amax(testa[t <= mate], at))
        # # plt.plot(t[t <= mate]/h/d, np.mean(testa[t <= mate], at))
        # # plt.title(tit)
        # # plt.ylim(-1, 121)
        # # plt.show()
        ################
        
        epoch = epoch + mates[cycle]

#%%# Population Proportion Data! Careful Use!

experiments = list(range(2))
experiments[0] = data.copy()
# experiments[1] = data.copy()

#%%# Vara Test

a = np.array(list(vara[0].values()))
b = np.array(list(vara[1].values()))
print(b/a)

#%%# Population Proportion Estimation!

safe = False
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Data'

epoch = 0
# simas = 10 # See above!
cycles = 8 # See above!

keys = [(key_0, key_1) for key_0 in range(simas) for key_1 in range(cycles)]
cos = ['tab:red', 'tab:olive', 'tab:cyan', 'tab:purple']

bids = [1, 1/2] # [True, False] # Experiments!

for cycle in range(cycles):
    alps = np.array([experiments[0][key]/np.sum(experiments[0][key], axis = 1).reshape(-1, 1) for key in keys if key[1] == cycle])
    bets = np.array([experiments[1][key]/np.sum(experiments[0][key], axis = 1).reshape(-1, 1) for key in keys if key[1] == cycle])
    ties = alps.shape[1] # ties = bets.shape[1]
    for t in range(ties):
        alp = {'Mean': np.mean(alps[:, t, :], axis = 0), 'Stand': np.std(alps[:, t, :], axis = 0)}
        bet = {'Mean': np.mean(bets[:, t, :], axis = 0), 'Stand': np.std(bets[:, t, :], axis = 0)}
        #
        tit = f'Cycle = {cycle+1}\nInter-Division Time = {mates[cycle]} Hours\nTime = [{np.round(epoch/24, 1)}, {np.round((epoch + mates[cycle])/24, 1)}] Days\n{t}'
        fit = f'Embryos = {reps} Cycle = {cycle+1}@{t} Shutdown = {any(shutdowns)}'
        fig, (axe, axi) = plt.subplots(1, 2, constrained_layout = True)
        fig.suptitle(tit)
        #
        x = list(range(1, 5))
        tick_label = ['N-G-', 'N+G-', 'N-G+', 'N+G+']
        #
        height = alp['Mean']
        axe.set_aspect('auto')
        axe.set_xlim(0, 5)
        axe.set_ylim(0, 1)
        axe.set_xlabel('Class')
        axe.set_title(f'Binomial? {bids[0]}')
        axe.set_ylabel('Proportion')
        axe.bar(x = x, height = height, yerr = alp['Stand'], tick_label = tick_label, color = cos)
        axe.grid(b = True, axis = 'y', color = 'lavender', linestyle = 'dashed')
        #
        height = bet['Mean']
        axi.set_aspect('auto')
        axi.set_xlim(0, 5)
        axi.set_ylim(0, 1)
        axi.set_xlabel('Class')
        axi.set_ylabel('Proportion')
        axi.set_title(f'Binomial? {bids[1]}')
        axi.bar(x = x, height = height, yerr = bet['Stand'], tick_label = tick_label, color = cos)
        axi.grid(b = True, axis = 'y', color = 'lavender', linestyle = 'dashed')
        #
        if safe:
            plt.savefig(path+'/'+fit+'.jpeg', dpi = 250)
        plt.show()
    epoch = epoch + mates[cycle]

#%%# Population Proportion Estimation! Cum!

safe = False
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Data'

epoch = 0
# simas = 10 # See above!
cycles = 8 # See above!

keys = [(key_0, key_1) for key_0 in range(simas) for key_1 in range(cycles)]
alp_cos = ['tab:red', 'tab:olive', 'tab:cyan', 'tab:purple']
bet_cos = ['tab:orange', 'tab:green', 'tab:pink', 'tab:brown']

bids = np.array([True, False]) # Experiments!
places = np.array(['Black', 'Gray'])

for cycle in range(cycles):
    alps = np.array([experiments[0][key]/np.sum(experiments[0][key], axis = 1).reshape(-1, 1) for key in keys if key[1] == cycle])
    bets = np.array([experiments[1][key]/np.sum(experiments[1][key], axis = 1).reshape(-1, 1) for key in keys if key[1] == cycle])
    ties = alps.shape[1] # ties = bets.shape[1]
    for t in range(ties):
        alp = [alps[:, t, fate] for fate in range(4)]
        bet = [bets[:, t, fate] for fate in range(4)]
        #
        tit = f'Cycle = {cycle+1}\nInter-Division Time = {mates[cycle]} Hours\nTime = [{np.round(epoch/24, 1)}, {np.round((epoch + mates[cycle])/24, 1)}] Days\nBinomial? <1: {places[bids].take(0)}, 0.5: {places[np.invert(bids)].take(0)}>\n{t}'
        fit = f'Cum Embryos = {reps} Cycle = {cycle+1}@{t} Shutdown = {any(shutdowns)}'
        fig, axe = plt.subplots(1, 4, constrained_layout = True)
        fig.suptitle(tit)
        #
        tick_label = ['N-G-', 'N+G-', 'N-G+', 'N+G+']
        #
        for fate in range(4):
            kist = stats.ks_2samp(data1 = alp[fate], data2 = bet[fate], mode = 'exact')
            axe[fate].set_title(f'Statistic\n{np.round(kist[0], 2)}\nP-Value\n{np.round(kist[1], 2)}')
            #
            axe[fate].set_aspect('equal')
            axe[fate].set_xlim(-0.05, 1.05)
            axe[fate].set_ylim(-0.05, 1.05)
            axe[fate].set_xlabel(f'{tick_label[fate]}')
            if fate == 0:
                axe[fate].set_ylabel('CDF')
            else:
                pass
            axe[fate].hist(x = alp[fate], bins = 100, range = (0, 1), density = True, cumulative = True, histtype = 'step', color = 'black')
            axe[fate].hist(x = bet[fate], bins = 100, range = (0, 1), density = True, cumulative = True, histtype = 'step', color = 'gray')
            axe[fate].vlines(x = [np.mean(alp[fate]), np.mean(bet[fate])], ymin = 0, ymax = 1, colors = [alp_cos[fate], bet_cos[fate]], linestyles = 'solid')
            x = [np.mean(alp[fate])-np.std(alp[fate]), np.mean(alp[fate])+np.std(alp[fate]), np.mean(bet[fate])-np.std(bet[fate]), np.mean(bet[fate])+np.std(bet[fate])]
            axe[fate].vlines(x = x, ymin = 0, ymax = 1, colors = [alp_cos[fate], alp_cos[fate], bet_cos[fate], bet_cos[fate]], linestyles = 'dotted')
            axe[fate].grid(b = True, axis = 'y', color = 'lavender', linestyle = 'dashed')
        #
        if safe:
            plt.savefig(path+'/'+fit+'.jpeg', dpi = 250)
        plt.show()
    epoch = epoch + mates[cycle]
