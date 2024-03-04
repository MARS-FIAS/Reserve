###########################################
######## Optimistic Initial Values ########
###########################################

#%%# Catalyzer

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numba
import math

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Multi-Armed Bandits [Action-Value Methods]

class BanditArm: # Arm := Action # (a | A)
    
    def __init__(self, action_value_true, optimistic_initial_value, seed = None):
        self.action_value_true = action_value_true # q := True Action Value := Expected/Mean Reward | Action Selection
        self.action_value_estimate = optimistic_initial_value # Q := Estimated Action Value
        self.N = 0 # Careful! Check "update_fun" comment! # Action/Arm Selection Set Cardinality
        self.seed = seed
    
    def reward_fun(self):
        R = np.random.default_rng(seed = self.seed).random() < self.action_value_true # Binomial Distribution # R = np.random.default_rng(seed = self.seed).normal(loc = self.action_value_true, scale = 1) # Gaussian Distribution
        return R
    
    def update_fun(self, R):
        self.N += 1
        self.action_value_estimate = self.action_value_estimate + (R - self.action_value_estimate) / self.N # ((self.N - 1) * self.action_value_estimate + R) / self.N
        return self

def experiment_fun(action_value_true_vet, optimistic_initial_values, T, seed = None): # Time Steps := Action Selections
    
    bandit_arms = [BanditArm(action_value_true_vet[j], optimistic_initial_values[j]) for j in range(len(action_value_true_vet))]
    
    rewards = np.zeros(T) # T := Final Time Step + 1 := Time Steps Number
    
    optimal_action = np.argmax([bandit_arm.action_value_true for bandit_arm in bandit_arms])
    print('Optimal Action:\t', optimal_action)
    
    for i in range(T): # i ~ t
        
        # Use Optimistic Initial Values
        
        j = np.argmax([bandit_arm.action_value_estimate for bandit_arm in bandit_arms])
        
        R = bandit_arms[j].reward_fun() # Reward Function
        rewards[i] = R # Rewards Log
        bandit_arms[j].update_fun(R) # Update Action/Arm Value Estimate
    
    for bandit_arm in bandit_arms:
        print('Action Value Estimate:\t', bandit_arm.action_value_estimate, '\t', bandit_arms.index(bandit_arm))
    
    # Calculate Statistics!
    
    stats = {'Total Reward Earned': rewards.sum(), 'Overall Win Rate': rewards.sum()/T}
    stats.update({f'Steps Bandit Arm {bandit_arms.index(bandit_arm)}': bandit_arm.N for bandit_arm in bandit_arms})
    for key in stats:
        value = stats.get(key)
        print('\n', key, '\n\t', value, sep = '')
    
    # Plot Statistics!
    
    win_rates = np.cumsum(rewards) / (np.arange(T) + 1)
    
    cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
    plt.title('Optimistic Initial Values = ' + str(optimistic_initial_values))
    plt.xlabel('Time Steps')
    plt.ylabel('Win Rates')
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid(alpha = 0.25, linestyle = '--')
    plt.hlines(y = action_value_true_vet, xmin = 0, xmax = T, colors = cocos[0:len(action_value_true_vet)], linestyles = '--')
    plt.plot(win_rates)
    plt.show()
    
    return win_rates

if __name__ == '__main__':
    
    action_value_true_vet = [0.25, 0.5, 0.75]
    optimistic_initial_values = [10, 10, 10]
    T = 10000
    
    experiment_fun(action_value_true_vet, optimistic_initial_values, T)
