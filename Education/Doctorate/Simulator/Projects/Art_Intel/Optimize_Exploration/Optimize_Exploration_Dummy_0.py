############################################
######## Optimize Exploration Dummy ########
############################################

#%%# Catalyzer

sup_comp = False # Super Computer?
inference_prod_activate = False # Activate Inference Procedure?
data_path = 'Optimize_Exploration_Dummy_0'
act = 0 # [0, 1, ...]
observe = 0 # {0, 1, ...}
curb = '' # {'Uncountable', 'Countable'}
restrict = {
    'Uncountable': 0,
    'Countable': 1
}

import sys
if sup_comp:
    path = '/home/biochemsim/ramirez/mars_projects/optimize_exploration/resources/'
else:
    path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/Art_Intel/Optimize_Exploration/Resources/'
sys.path.append(path)
import numpy as np
import torch
import time
import pickle
if not sup_comp:
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 250
    plt.rcParams['savefig.dpi'] = 250

#%%# Simulation [Preparation]

from Utilities_Optimize_Exploration import make_paras

para_set_raw = {
    'Mean_0': (10, (-10, 30), restrict[curb], None), 'Mean_1': (10, (-10, 30), restrict[curb], None),
    'Variance_0': (4, (1, 10), restrict[curb], None), 'Variance_1': (4, (1, 10), restrict[curb], None)
}
para_set_mode = 0 # {0, 1} # {'No Remap', 'Remap: [A, B] ---->>>> [0, 1]'}
para_set, para_set_true = make_paras(para_set_raw = para_set_raw, para_set_mode = para_set_mode, verbose = True)

#%%# Simulation [Functions]

def simulator_dummy_zero(parameter_set, data_points = 2500):
    let = int(parameter_set.size(0)/2)
    mean_vet = parameter_set[[index for index in range(let) if index < let]]
    _covariance_mat = parameter_set[[index for index in range(2*let) if index > let-1]]
    covariance_mat = torch.eye(let)*_covariance_mat
    normality = torch.distributions.MultivariateNormal(loc = mean_vet, covariance_matrix = covariance_mat)
    trajectory_set = normality.sample(tuple([data_points]))
    return trajectory_set

def objective_fun_dummy_zero(trajectory_set, boxes = 10, mean_aim = 0, variance_aim = 1, **keywords):
    let = int(trajectory_set.size(0)/boxes)
    score_set = torch.zeros(boxes)
    alp = keywords.get('alp', 0.5)
    bet = keywords.get('bet', 1-alp)
    mess = f'Oops! Something went wrong!\n{alp+bet} != 1'
    check = alp+bet == 1
    assert check, mess
    _score_set = torch.sum(input = trajectory_set, dim = 1)
    zero = torch.tensor(0)
    for box in range(boxes):
        i = let*box
        j = let*(box+1)
        mess = 'Oops! Something went wrong!'
        check = j-i == let
        assert check, mess
        mean_score = torch.max(1-torch.abs(mean_aim-torch.mean(_score_set[i:j])), zero) # (L1)^1
        variance_score = torch.max(1-torch.pow(variance_aim-torch.var(_score_set[i:j]), 2), zero) # (L2)^2
        score_set[box] = alp*mean_score+bet*variance_score
    return score_set

def appraise_dummy_zero(score_set, quants = torch.tensor([0.05, 0.5, 0.95]), verbose = False, **keywords):
    phi = keywords.get('phi', 1-0.5**3)
    psi = keywords.get('psi', 1-phi)
    mess = f'Oops! Something went wrong!\n{phi+psi} != 1'
    check = phi+psi == 1
    assert check, mess
    quantiles = torch.quantile(input = score_set, q = quants, dim = 0, interpolation = 'nearest')
    if quantiles.dim() == 1:
        quantiles = torch.reshape(quantiles, (-1, 1))
    alp = quantiles[1, :]
    bet = 1-torch.abs(quantiles[2, :]-quantiles[0, :])
    chi = phi*alp+psi*bet
    meta_score = torch.mean(chi)
    if verbose:
        simul_trials = score_set.size(0)
        x_mini, x_maxi = 0, score_set.size(1)
        y_mini, y_maxi = 0, 1
        cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
        for simul_trial in range(simul_trials):
            coco = cocos[simul_trial % len(cocos)]
            plt.plot(score_set[simul_trial], color = coco, alpha = 0.25)
        plt.plot(quantiles[0, :], linestyle = '--', color = 'r')
        plt.plot(quantiles[1, :], linestyle = '--', color = 'g')
        plt.plot(quantiles[2, :], linestyle = '--', color = 'b')
        plt.plot(chi, linestyle = '-', color = 'm')
        plt.axhline(meta_score, x_mini, x_maxi, color = 'k', alpha = 0.5)
        plt.xlim(x_mini, x_maxi)
        plt.ylim(y_mini, y_maxi)
        plt.grid(linestyle = '--', alpha = 0.25)
        plt.show()
    return meta_score

#%%# Simulation [Arguments]

parameter_set = torch.tensor(data = para_set, dtype = torch.float32)
data_points = int(250e3) # [250e1, 250e4]

# trajectory_set
boxes = 100 # [10, 1000]
mean_aim = 2*10 # Mean_Aim = Mean_2 = Mean_0 + Mean_1
variance_aim = 2*4 # Variance_Aim = Variance_2 = Variance_0 + Variance_1
alp = 0.5
bet = 1-alp

# score_set
quants = torch.tensor([0.05, 0.5, 0.95])
verbose = not sup_comp
phi = 1-0.5**3
psi = 1-phi

seed = None

#%%# Simulation [Local Computer Test]

simul_trials = 25

score_set = torch.full((simul_trials, boxes), torch.nan)

if not sup_comp:
    for simul_trial in range(simul_trials):
        trajectory_set = simulator_dummy_zero(parameter_set, data_points)
        score_set[simul_trial] = objective_fun_dummy_zero(trajectory_set, boxes, mean_aim, variance_aim, alp = alp, bet = bet)
    meta_score = appraise_dummy_zero(score_set, quants, verbose)

#%%# Inference Procedure [Preparation]

from Utilities_Twin import make_prior
from Utilities_Optimize_Exploration import make_prior_mixer

para_set_sieve = None
verbose = not sup_comp

prior = make_prior(para_set_true, para_set_mode, para_set_sieve, verbose)
prior_mixer = make_prior_mixer(para_set_true, para_set_mode, para_set_sieve, verbose)

#%%# Simulation /\ Inference Procedure!

tag = f'Act_{act}_Observe_{observe}_{curb.capitalize()}'
if sup_comp:
    path = f'/scratch/biochemsim/ramirez/mars_projects/optimize_exploration/data_bank/{data_path}/'
else:
    path = f'/media/mars-fias/MARS/MARS_Data_Bank/World/Optimize_Exploration/{data_path}/'
post = '_Posterior.pkl'

#%%# Simulation Procedure: Synopsis Preparation!

def make_synopsis(explorations, para_set_true, verbose = False):
    synopsis_varas = list(para_set_true.keys())+['Register'] # {'Register': {'Current_Meta_Score', 'Optimum_Meta_Score'}}
    synopsis_size = (2, explorations, len(synopsis_varas)) # {0: 'Current_Meta_Score', 1: 'Optimum_Meta_Score'}
    synopsis = torch.full(synopsis_size, torch.nan)
    if verbose: print(f'Synopsis!\t{synopsis.size()}')
    return synopsis

def refresh_synopsis(synopsis, exploration, roster_varas = tuple(), verbose = False):
    current_para_set, current_meta_score, optimum_para_set, optimum_meta_score = roster_varas
    synopsis[0, exploration, :] = torch.cat((current_para_set, current_meta_score.reshape(tuple([1]))))
    synopsis[1, exploration, :] = torch.cat((optimum_para_set, optimum_meta_score.reshape(tuple([1]))))
    return synopsis

def save_synopsis(synopsis):
    return None

#%%# Simulation Procedure: Optimize Exploration!

from Utilities_Optimize_Exploration import instate_para_set, move_para_set

if not inference_prod_activate:
    
    safe = False
    verbose = False
    if sup_comp: # Super Computer
        task_pin = int(sys.argv[1])
        tasks = 10
        task_explorations = 2*250 # task_simulations
    else: # Local Computer
        task_pin = 0
        tasks = 2
        task_explorations = 2*250 # task_simulations
    simul_trials = 25
    if restrict[curb]: # 'Countable'
        proposal = prior_mixer
        _prior_mixer = prior_mixer
    else: # 'Uncountable'
        proposal = prior
        _prior_mixer = None
    iota = 0.025
    alpha_dit = torch.distributions.Uniform(0, 1)
    initial_temp = 10
    spa = ' '*8
    
    for task in range(tasks):
        print('~>'*8)
        initial_para_set = instate_para_set(proposal)
        synopsis = make_synopsis(task_explorations, para_set_true, verbose)
        for exploration in range(task_explorations):
            if exploration == 0:
                current_para_set = initial_para_set
                for simul_trial in range(simul_trials):
                    trajectory_set = simulator_dummy_zero(current_para_set, data_points)
                    score_set[simul_trial] = objective_fun_dummy_zero(trajectory_set, boxes, mean_aim, variance_aim)
                current_meta_score = appraise_dummy_zero(score_set, quants, verbose)
                optimum_para_set = current_para_set
                optimum_meta_score = current_meta_score
                roster_varas = (current_para_set, current_meta_score, optimum_para_set, optimum_meta_score)
                synopsis = refresh_synopsis(synopsis, exploration, roster_varas, verbose)
            else:
                explore_para_set = move_para_set(current_para_set, iota, prior, _prior_mixer, verbose)
                for simul_trial in range(simul_trials):
                    trajectory_set = simulator_dummy_zero(explore_para_set, data_points)
                    score_set[simul_trial] = objective_fun_dummy_zero(trajectory_set, boxes, mean_aim, variance_aim)
                explore_meta_score = appraise_dummy_zero(score_set, quants, verbose)
                delta_meta_score = explore_meta_score - current_meta_score
                alpha = alpha_dit.sample()
                temp = initial_temp/exploration # exploration != 0
                criterion = alpha < torch.exp(delta_meta_score/temp)
                condition = delta_meta_score >= 0 or criterion
                if condition:
                    current_para_set = explore_para_set
                    current_meta_score = explore_meta_score
                if explore_meta_score > optimum_meta_score:
                    optimum_para_set = explore_para_set
                    optimum_meta_score = explore_meta_score
                    print(f'{task_pin}{spa}{task}{spa}{optimum_meta_score}\n{spa}{optimum_para_set.tolist()}')
                roster_varas = (explore_para_set, explore_meta_score, optimum_para_set, optimum_meta_score)
                synopsis = refresh_synopsis(synopsis, exploration, roster_varas, verbose)
        print('<~'*8)

#%%# Inference Procedure: Simul Data Load! [Super Computer ---->>>> Local Computer]



#%%# Inference Procedure: Simul Data Plot! [Super Computer ---->>>> Local Computer]



#%%# Inference Procedure: Optimize Exploration!



#%%# Inference Procedure: Posterior Data Save!

if inference_prod_activate:
    posterior = posts[where]
    _tag = f'Act_{act}_Observe_{_observe}_{curb.capitalize()}'
    if act == 0:
        locus = path + f'Observe_{_observe}/' + _tag + post
    else:
        locus = path + _tag + post
    with open(locus, 'wb') as portfolio:
        pickle.dump(posterior, portfolio)
