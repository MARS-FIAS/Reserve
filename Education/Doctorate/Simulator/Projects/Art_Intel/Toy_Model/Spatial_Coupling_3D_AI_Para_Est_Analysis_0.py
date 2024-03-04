##########################################################
######## Spatial Coupling 3D AI Para Est Analysis ########
##########################################################

#%%# Catalyzer

# import sys
import os
import re
# import time

import numpy as np
# import torch
# import sbi

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
# plt.rcParams['figure.figsize'] = plt.figaspect(1)

#%%# Analysis Procedure: Fun!

def analysis_prod_fun(path, selection, take):
    
    # Analysis Procedure [Preparation]
    
    # path = '/media/mars-fias/MARS/MARS_Data_Bank/Art_Intel/Spatial_Coupling_3D_AI_Para_Est_0/'
    if not os.path.exists(path):
        mess = f"Oops! The file directory '{path}' is invalid!"
        raise RuntimeError(mess)
    
    _tags = os.listdir(path)
    l = 'Data_'
    r = '.npz'
    pat = f'({l}|{r})'
    rec = ''
    tags = []
    for _tag in _tags:
        tag = re.sub(pat, rec, _tag)
        tags.append(tag)
    
    _targets = set()
    pat = '(_\d)$'
    rec = ''
    for tag in tags:
        target = re.sub(pat, rec, tag)
        _targets.add(target)
    targets = list(_targets)
    mess = f'Available data files!\n\t{targets}'
    print(mess)
    
    _targets_decrypt = []
    pat = '_'
    decrypt = '[[ScopeAlp_ScopeBet], [ParaSetKeys], [ParaSetValues]]'
    for target in targets:
        _target = re.split(pat, target, 1)
        vara = _target[0]
        refresh = _target[1]
        if refresh in ['None', 'Prior']:
            include_keys = ['N_N', 'G_G', 'FI_N', 'G_EA', 'G_N', 'N_G', 'FI_G', 'N_EA']
            include_values = [1]*len(include_keys)
        else:
            include_keys = ['N_N', 'G_N', 'N_G']
            include_values = [1]*len(include_keys)
        pat_spot = '-'
        spot = re.findall(pat_spot, vara)
        if len(spot) == 1:
            include_values[include_keys.index('N_EA')] = 0.5
        _targets_decrypt.append([_target, include_keys, include_values])
    
    targets_decrypt_temp = {targets[index]: _targets_decrypt[index] for index in range(len(targets))}
    # selection = ['1000_None', '1000-500_None'] # ['1000-500_None', '750_400', '750_800', '750_None', '1250_400', '1250_800', '1000_None', '500_None']
    for pick in selection:
        if pick not in targets_decrypt_temp.keys():
            mess = f"The pick '{pick}' is invalid!"
            raise RuntimeError(mess)
    mess = f'\nOur selection!\n\t{selection}'
    print(mess)
    targets_decrypt = {pick: targets_decrypt_temp[pick] for pick in selection}
    
    # Analysis Procedure [Data Load]
    
    _take = ['median', 'mape', 'posterior_samples'] # Only possible choices!
    # take = ['mape']
    for use in take:
        if use not in _take:
            mess = f"The use of '{use}' is not possible!"
            raise RuntimeError(mess)
    mess = f'\nOur take!\n\t{take}'
    print(mess)
    
    data = {pick: None for pick in selection}
    for pick in selection:
        pick_tags = [tags[index] for index in range(len(tags)) if len(re.findall(f'^({pick})(_\d)$', tags[index])) == 1]
        for pick_tag in pick_tags:
            pick_data = np.load(path+l+pick_tag+r)
            if pick_tags.index(pick_tag) == 0:
                for use in take:
                    use_data = {use: None for use in take}
                    use_data[use] = pick_data[use]
            else:
                for use in take:
                    use_data[use] = np.concatenate((use_data[use], pick_data[use]))
        data[pick] = use_data
    
    ret = (data, targets_decrypt, decrypt)
    
    return ret

#%%# MAPE Data Analysis [Data Collection]

path = '/media/mars-fias/MARS/MARS_Data_Bank/Art_Intel/Spatial_Coupling_3D_AI_Para_Est_0/'
selection = ['500_None', '750_None', '1000_None']
take = ['mape']
data, targets_decrypt, decrypt = analysis_prod_fun(path, selection, take)
use = take[0]

#%%# 'Pairwise /\ Equal Parameters /\ Different Databases'

theta = 'N_EA' # ['N_N', 'G_G', 'FI_N', 'G_EA', 'G_N', 'N_G', 'FI_G', 'N_EA']
use = 'mape'
bin_step = 5*5

theta_x = theta
theta_y = theta

plt.rcParams['figure.figsize'] = plt.figaspect(0.5)
cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
fig, axe = plt.subplots(1, 2, constrained_layout = True)
_tit = 'Pairwise /\ Equal Parameters /\ Different Databases'
tit = _tit+' /\ '+'Joint'
fig.suptitle(tit)

fates = list(data.keys())
pairs = [(index, index+1) for index in range(0, len(fates)-1)]

signs = ['.', '+', '*', '8', 'D']
limits = (0, 1000)
step = 125
labels = [fates[pair[0]]+' '+fates[pair[1]] for pair in pairs]

axe[0].set_aspect('equal')
axe[0].set_title(theta_x)
axe[0].set_xlim(limits)
axe[0].set_ylim(limits)
axe[0].set_xticks(np.arange(limits[0], limits[1]+step, step))
axe[0].set_yticks(np.arange(limits[0], limits[1]+step, step))
# axe[0].set_xlabel(theta_x)
# axe[0].set_ylabel(theta_y)

verticals = []
horizontals = []
for pair in pairs:
    index = pairs.index(pair)
    fate_x = fates[pair[0]]
    fate_y = fates[pair[1]]
    _theta_x = targets_decrypt[fate_x][1].index(theta_x)
    _theta_y = targets_decrypt[fate_y][1].index(theta_y)
    x = data[fate_x][use][:, _theta_x]
    y = data[fate_y][use][:, _theta_y]
    axe[0].scatter(x, y, marker = signs[pairs.index(pair)], color = cocos[index], label = labels[index])
    try:
        vertical = int(targets_decrypt[fate_x][0][0])/2
        horizontal = int(targets_decrypt[fate_y][0][0])/2
    except:
        vertical = int(re.sub('(-\d+)$', '', targets_decrypt[fate_x][0][0]))/2
        horizontal = int(re.sub('(-\d+)$', '', targets_decrypt[fate_y][0][0]))/2
    verticals.append(vertical)
    horizontals.append(horizontal)
    axe[0].vlines(vertical, limits[0], limits[1], colors = cocos[index], linestyles = '--', alpha = 0.125)
    axe[0].hlines(horizontal, limits[0], limits[1], colors = cocos[index], linestyles = '--', alpha = 0.125)

for pair in pairs:
    index = pairs.index(pair)
    fate_x = fates[pair[0]]
    fate_y = fates[pair[1]]
    _theta_x = targets_decrypt[fate_x][1].index(theta_x)
    _theta_y = targets_decrypt[fate_y][1].index(theta_y)
    x = data[fate_x][use][:, _theta_x]
    y = data[fate_y][use][:, _theta_y]
    axe[0].scatter(np.median(x), np.median(y), marker = signs[index], color = cocos[::-1][index])
    axe[0].vlines(np.median(x), np.min(y), np.max(y), colors = cocos[::-1][index], alpha = 0.5)
    axe[0].hlines(np.median(y), np.min(x), np.max(x), colors = cocos[::-1][index], alpha = 0.5)
    me = [np.round(np.median(y)/np.median(x), 5), np.round(np.median(x)/np.median(y), 5)]
    ex = [np.round(horizontals[index]/verticals[index], 3), np.round(verticals[index]/horizontals[index], 3)]
    print(f"Ratio\n\t'{fate_y}/{fate_x}'\t{me[0]}\t{ex[0]}\n\t'{fate_x}/{fate_y}'\t{me[1]}\t{ex[1]}")
axe[0].legend(framealpha = 0.125, fontsize = 'xx-small')

axe[1].set_aspect('equal')
axe[1].set_title(theta_y)
axe[1].set_xlim(limits)
axe[1].set_ylim(limits)
axe[1].set_xticks(np.arange(limits[0], limits[1]+step, step))
axe[1].set_yticks(np.arange(limits[0], limits[1]+step, step))
# axe[1].set_xlabel(theta_x)
# axe[1].set_ylabel(theta_y)

_hist_x = [fate for fate in range(0, len(fates)-1)]
_hist_y = [fate for fate in range(1, len(fates))]

verticals = []
horizontals = []
for index in range(len(fates)-1):
    fate_x = fates[_hist_x[index]]
    fate_y = fates[_hist_y[index]]
    _theta_x = targets_decrypt[fate_x][1].index(theta_x)
    _theta_y = targets_decrypt[fate_y][1].index(theta_y)
    if index == 0:
        hist_x = data[fate_x][use][:, _theta_x]
        hist_y = data[fate_y][use][:, _theta_y]
        # print(fate_x, fate_y)
    else:
        hist_x = np.concatenate((hist_x, data[fate_x][use][:, _theta_x]))
        hist_y = np.concatenate((hist_y, data[fate_y][use][:, _theta_y]))
        # print(fate_x, fate_y)
    try:
        vertical = int(targets_decrypt[fate_x][0][0])/2
        horizontal = int(targets_decrypt[fate_y][0][0])/2
    except:
        vertical = int(re.sub('(-\d+)$', '', targets_decrypt[fate_x][0][0]))/2
        horizontal = int(re.sub('(-\d+)$', '', targets_decrypt[fate_y][0][0]))/2
    verticals.append(vertical)
    horizontals.append(horizontal)
axe[1].hist2d(hist_x, hist_y, bins = int(limits[1]/bin_step), range = [limits]*2)
axe[1].vlines(verticals, limits[0], limits[1], colors = 'tab:gray', linestyles = '--', alpha = 0.25)
axe[1].hlines(horizontals, limits[0], limits[1], colors = 'tab:gray', linestyles = '--', alpha = 0.25)
plt.show()

plt.rcParams['figure.figsize'] = plt.figaspect(1)
fig, axe = plt.subplots(1, 1, constrained_layout = True)
tit = _tit+' /\ '+'Marginal'
fig.suptitle(tit)

axe.set_title(theta_x)
axe.set_xticks(np.arange(limits[0], limits[1]+step, step))

for fate in fates:
    x = data[fate][use][:, _theta_x]
    axe.hist(x, range = limits, bins = int(limits[1]/bin_step), color = cocos[::-1][fates.index(fate)], density = True, cumulative = False, label = fate, alpha = 1-fates.index(fate)*0.125)
    axe.axvline(np.median(x), color = cocos[fates.index(fate)], linestyle = '--', alpha = 0.5)
axe.legend(framealpha = 0.125, fontsize = 'xx-small')
plt.show()

#%%# Pairwise /\ Equal|Different Parameters /\ Equal Databases

theta_x = 'G_EA' # ['N_N', 'G_G', 'FI_N', 'G_EA', 'G_N', 'N_G', 'FI_G', 'N_EA']
theta_y = 'N_EA' # ['N_N', 'G_G', 'FI_N', 'G_EA', 'G_N', 'N_G', 'FI_G', 'N_EA']
use = 'mape'
bin_step = 5*5

plt.rcParams['figure.figsize'] = plt.figaspect(0.5)
cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
fig, axe = plt.subplots(1, 2, constrained_layout = True)
_tit = 'Pairwise /\ Equal|Different Parameters /\ Equal Databases'
tit = _tit+' /\ '+'Joint'
fig.suptitle(tit)

fates = list(data.keys())

signs = ['.', '+', '*', '8', 'D']
limits = (0, 1000)
step = 125
labels = fates # labels = [f'{theta_x} {theta_y} {fate}' for fate in fates]

axe[0].set_aspect('equal')
axe[0].set_xlim(limits)
axe[0].set_ylim(limits)
axe[0].set_xticks(np.arange(limits[0], limits[1]+step, step))
axe[0].set_yticks(np.arange(limits[0], limits[1]+step, step))
axe[0].set_xlabel(theta_x)
axe[0].set_ylabel(theta_y)

for fate in fates:
    index = fates.index(fate)
    _theta_x = targets_decrypt[fate][1].index(theta_x)
    _theta_y = targets_decrypt[fate][1].index(theta_y)
    x = data[fate][use][:, _theta_x]
    y = data[fate][use][:, _theta_y]
    axe[0].scatter(x, y, marker = signs[index], color = cocos[index], label = labels[index])
    try:
        vertical = int(targets_decrypt[fate][0][0])/2
        horizontal = int(targets_decrypt[fate][0][0])/2
    except:
        vertical = int(re.sub('(-\d+)$', '', targets_decrypt[fate][0][0]))/2
        horizontal = int(re.sub('(-\d+)$', '', targets_decrypt[fate][0][0]))/2
    axe[0].vlines(vertical, limits[0], limits[1], colors = cocos[index], linestyles = '--', alpha = 0.125)
    axe[0].hlines(horizontal, limits[0], limits[1], colors = cocos[index], linestyles = '--', alpha = 0.125)

for fate in fates:
    index = fates.index(fate)
    _theta_x = targets_decrypt[fate][1].index(theta_x)
    _theta_y = targets_decrypt[fate][1].index(theta_y)
    x = data[fate][use][:, _theta_x]
    y = data[fate][use][:, _theta_y]
    axe[0].scatter(np.median(x), np.median(y), marker = signs[index], color = cocos[::-1][index])
    axe[0].vlines(np.median(x), np.min(y), np.max(y), colors = cocos[::-1][index], alpha = 0.5)
    axe[0].hlines(np.median(y), np.min(x), np.max(x), colors = cocos[::-1][index], alpha = 0.5)
    print(f"Ratio {fate}\n\t'{theta_y}/{theta_x}'\t{np.round(np.median(y)/np.median(x), 5)}\n\t'{theta_x}/{theta_y}'\t{np.round(np.median(x)/np.median(y), 5)}")
axe[0].legend(framealpha = 0.125, fontsize = 'xx-small')

axe[1].set_aspect('equal')
axe[1].set_xlim(limits)
axe[1].set_ylim(limits)
axe[1].set_xticks(np.arange(limits[0], limits[1]+step, step))
axe[1].set_yticks(np.arange(limits[0], limits[1]+step, step))
axe[1].set_xlabel(theta_x)
axe[1].set_ylabel(theta_y)

verticals = []
horizontals = []
for fate in fates:
    index = fates.index(fate)
    _theta_x = targets_decrypt[fate][1].index(theta_x)
    _theta_y = targets_decrypt[fate][1].index(theta_y)
    x = data[fate][use][:, _theta_x]
    y = data[fate][use][:, _theta_y]
    if index == 0:
        hist_x = x
        hist_y = y
    else:
        hist_x = np.concatenate((hist_x, x))
        hist_y = np.concatenate((hist_y, y))
    try:
        vertical = int(targets_decrypt[fate][0][0])/2
        horizontal = int(targets_decrypt[fate][0][0])/2
    except:
        vertical = int(re.sub('(-\d+)$', '', targets_decrypt[fate][0][0]))/2
        horizontal = int(re.sub('(-\d+)$', '', targets_decrypt[fate][0][0]))/2
    verticals.append(vertical)
    horizontals.append(horizontal)
axe[1].hist2d(hist_x, hist_y, bins = int(limits[1]/bin_step), range = [limits]*2)
axe[1].vlines(verticals, limits[0], limits[1], colors = 'tab:gray', linestyles = '--', alpha = 0.25)
axe[1].hlines(horizontals, limits[0], limits[1], colors = 'tab:gray', linestyles = '--', alpha = 0.25)
plt.show()

plt.rcParams['figure.figsize'] = plt.figaspect(1/len(fates))
fig, axe = plt.subplots(1, len(fates), constrained_layout = True)
tit = _tit+' /\ '+'Marginal'
fig.suptitle(tit)

for fate in fates:
    cocos_0 = [coco for coco in cocos if cocos.index(coco) % 2 == 0]
    cocos_1 = [coco for coco in cocos if cocos.index(coco) % 2 == 1]
    index = fates.index(fate)
    axe[index].set_title(fate)
    axe[index].set_xticks(np.arange(limits[0], limits[1]+step, step))
    x = data[fate][use][:, _theta_x]
    axe[index].hist(x, range = limits, bins = int(limits[1]/bin_step), color = cocos_0[::-1][fates.index(fate)], density = True, cumulative = False, label = theta_x, alpha = 1-fates.index(fate)*0.125)
    axe[index].axvline(np.median(x), color = cocos_0[fates.index(fate)], linestyle = '--', alpha = 0.5)
    y = data[fate][use][:, _theta_y]
    axe[index].hist(y, range = limits, bins = int(limits[1]/bin_step), color = cocos_1[::-1][fates.index(fate)], density = True, cumulative = False, label = theta_y, alpha = 1-fates.index(fate)*0.125)
    axe[index].axvline(np.median(y), color = cocos_1[fates.index(fate)], linestyle = '--', alpha = 0.5)
    axe[index].legend(framealpha = 0.125, fontsize = 'xx-small')
plt.show()

#%%# Rats AND Scores [Fun]

def rats_fun(theta_x, theta_y, data, targets_decrypt, use):
    fates = list(data.keys())
    rat_y_x = np.zeros(tuple([len(fates)]))
    rat_x_y = np.zeros(tuple([len(fates)]))
    for fate in fates:
        index = fates.index(fate)
        _theta_x = targets_decrypt[fate][1].index(theta_x)
        _theta_y = targets_decrypt[fate][1].index(theta_y)
        x = data[fate][use][:, _theta_x]
        y = data[fate][use][:, _theta_y]
        pre = 7
        rat_y_x[index] = np.round(np.median(y)/np.median(x), pre) # 'theta_y/theta_x'
        rat_x_y[index] = np.round(np.median(x)/np.median(y), pre) # 'theta_x/theta_y'
    rat = {'y_x': rat_y_x, 'x_y': rat_x_y}
    # print(rat)
    return rat

def scores_fun(theta_x, theta_y, data, targets_decrypt, use):
    fates = list(data.keys())
    score_y_x = np.zeros(tuple([len(fates)]))
    score_x_y = np.zeros(tuple([len(fates)]))
    for fate in fates:
        index = fates.index(fate)
        _theta_x = targets_decrypt[fate][1].index(theta_x)
        _theta_y = targets_decrypt[fate][1].index(theta_y)
        x = data[fate][use][:, _theta_x]
        y = data[fate][use][:, _theta_y]
        pre = 7
        _x = np.mean(x)/np.std(x)
        _y = np.mean(y)/np.std(y)
        score_y_x[index] = np.round(_y/_x, pre) # 'theta_y/theta_x'
        score_x_y[index] = np.round(_x/_y, pre) # 'theta_x/theta_y'
    score = {'y_x': score_y_x, 'x_y': score_x_y}
    # print(score)
    return score

show = False

#%%# Rats! [Y/X AND X/Y]

thetas = ['N_N', 'G_G', 'FI_N', 'G_EA', 'G_N', 'N_G', 'FI_G', 'N_EA']
theta_x = 'N_EA'
_tit = ' ~ Median'
rats_y_x = np.zeros((len(thetas), len(fates)))
rats_x_y = np.zeros((len(thetas), len(fates)))

for theta_y in thetas:
    index = thetas.index(theta_y)
    rat = rats_fun(theta_x, theta_y, data, targets_decrypt, use)
    rats_y_x[index] = rat['y_x']
    rats_x_y[index] = rat['x_y']

# Plot Rats! [Y/X]

tit = f'Ratio\nY / {theta_x}'+_tit

if show:
    plt.rcParams['figure.figsize'] = plt.figaspect(1)
    for fate in fates:
        index = fates.index(fate)
        x = np.repeat(index, len(thetas))
        y = rats_y_x[:, index]
        plt.title(tit)
        plt.plot(x, y, linestyle = 'None', marker = '_')
    plt.xticks(range(len(fates)), fates)
    plt.xlim(-0.1, len(fates)-1+0.1)

plt.rcParams['figure.figsize'] = plt.figaspect(1)
fig, axe = plt.subplots(1, 1, constrained_layout = True)
fig.suptitle(tit)

for fate in fates:
    index = fates.index(fate)
    x = range(len(thetas))
    y = rats_y_x[:, index]
    axe.bar(x = x, height = y, width = 0.75-index*0.125, color = cocos[index], tick_label = thetas, alpha = 1-index*0.125, label = fate)
    axe.legend(framealpha = 0.125, fontsize = 'xx-small')
plt.show()

# Plot Rats! [X/Y]

tit = f'Ratio\n{theta_x} / Y'+_tit

if show:
    plt.rcParams['figure.figsize'] = plt.figaspect(1)
    for fate in fates:
        index = fates.index(fate)
        x = np.repeat(index, len(thetas))
        y = rats_x_y[:, index]
        plt.title(tit)
        plt.plot(x, y, linestyle = 'None', marker = '_')
    plt.xticks(range(len(fates)), fates)
    plt.xlim(-0.1, len(fates)-1+0.1)

plt.rcParams['figure.figsize'] = plt.figaspect(1)
fig, axe = plt.subplots(1, 1, constrained_layout = True)
fig.suptitle(tit)

for fate in fates:
    index = fates.index(fate)
    x = range(len(thetas))
    y = rats_x_y[:, index]
    axe.bar(x = x, height = y, width = 0.75-index*0.125, color = cocos[index], tick_label = thetas, alpha = 1-index*0.125, label = fate)
    axe.legend(framealpha = 0.125, fontsize = 'xx-small')
plt.show()

#%%# Scores! [Y/X AND X/Y]

thetas = ['N_N', 'G_G', 'FI_N', 'G_EA', 'G_N', 'N_G', 'FI_G', 'N_EA']
theta_x = 'N_EA'
_tit = ' ~ Mean/SD'
scores_y_x = np.zeros((len(thetas), len(fates)))
scores_x_y = np.zeros((len(thetas), len(fates)))

for theta_y in thetas:
    index = thetas.index(theta_y)
    score = scores_fun(theta_x, theta_y, data, targets_decrypt, use)
    scores_y_x[index] = score['y_x']
    scores_x_y[index] = score['x_y']

# Plot Scores! [Y/X]

tit = f'Ratio\nY / {theta_x}'+_tit

if show:
    plt.rcParams['figure.figsize'] = plt.figaspect(1)
    for fate in fates:
        index = fates.index(fate)
        x = np.repeat(index, len(thetas))
        y = scores_y_x[:, index]
        plt.title(tit)
        plt.plot(x, y, linestyle = 'None', marker = '_')
    plt.xticks(range(len(fates)), fates)
    plt.xlim(-0.1, len(fates)-1+0.1)
    
plt.rcParams['figure.figsize'] = plt.figaspect(1)
fig, axe = plt.subplots(1, 1, constrained_layout = True)
fig.suptitle(tit)

for fate in fates:
    index = fates.index(fate)
    x = range(len(thetas))
    y = scores_y_x[:, index]
    axe.bar(x = x, height = y, width = 0.75-index*0.125, color = cocos[index], tick_label = thetas, alpha = 1-index*0.125, label = fate)
    axe.legend(framealpha = 0.125, fontsize = 'xx-small')
plt.show()

# Plot Scores! [X/Y]

tit = f'Ratio\n{theta_x} / Y'+_tit

if show:
    plt.rcParams['figure.figsize'] = plt.figaspect(1)
    for fate in fates:
        index = fates.index(fate)
        x = np.repeat(index, len(thetas))
        y = scores_x_y[:, index]
        plt.title(tit)
        plt.plot(x, y, linestyle = 'None', marker = '_')
    plt.xticks(range(len(fates)), fates)
    plt.xlim(-0.1, len(fates)-1+0.1)

plt.rcParams['figure.figsize'] = plt.figaspect(1)
fig, axe = plt.subplots(1, 1, constrained_layout = True)
fig.suptitle(tit)

for fate in fates:
    index = fates.index(fate)
    x = range(len(thetas))
    y = scores_x_y[:, index]
    axe.bar(x = x, height = y, width = 0.75-index*0.125, color = cocos[index], tick_label = thetas, alpha = 1-index*0.125, label = fate)
    axe.legend(framealpha = 0.125, fontsize = 'xx-small')
plt.show()

#%%# Section [New]


