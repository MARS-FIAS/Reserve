#################################################
######## Bayesian Bandits Beta Bernoulli ########
#################################################

#%%# Catalyzer

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import beta
# import math

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Multi-Armed Bandits [Action-Value Methods]

class BanditArm: # Arm := Action # (a | A)
    
    def __init__(self, q, alp, bet, seed = None):
        self.q = q # q := True Action Value := Expected/Mean Reward | Action Selection
        self.alp = alp # Prior Distribution
        self.bet = bet # Prior Distribution
        self.N = 0 # Action/Arm Selection Set Cardinality
        self.seed = seed
    
    def reward_fun(self):
        R = np.random.default_rng(seed = self.seed).random() < self.q # Binomial Distribution
        return R
    
    def sample_fun(self):
        Q = np.random.default_rng(seed = self.seed).beta(self.alp, self.bet, 1) # Q := Estimated Action Value
        return Q
    
    def update_fun(self, R):
        self.alp += R
        self.bet += 1 - R
        self.N += 1
        return self

def draw_fun(bandit_arms, t):
    x = np.linspace(0, 1, 251)
    for bandit_arm in bandit_arms:
        y = beta.pdf(x, bandit_arm.alp, bandit_arm.bet)
        plt.plot(x, y, label = f'True Action Value q = {bandit_arm.q} | Win Rate = {bandit_arm.alp-1}/{bandit_arm.N}')
    plt.title(f'Bandit-Arm Posterior Distribution\nTime Steps = {t}')
    plt.legend()
    plt.grid(alpha = 0.25, linestyle = '--')
    plt.show()
    return None

def experiment_fun(q_vet, T, steps_draw, seed = None): # Time Steps := Action Selections
    
    bandit_arms = [BanditArm(q, 1, 1) for q in q_vet]
    
    rewards = np.zeros(T) # T := Final Time Step + 1 := Time Steps Number
    
    optimal_action = np.argmax([bandit_arm.q for bandit_arm in bandit_arms])
    print('Optimal Action:\t', optimal_action)
    
    for i in range(T): # i ~ t
        
        # Use Thompson Sampling
        
        j = np.argmax([bandit_arm.sample_fun() for bandit_arm in bandit_arms])
        
        # Plot Statistics!
        
        if i in steps_draw:
            draw_fun(bandit_arms, i)
        
        # Reward Function | Rewards Log | Update Action/Arm Value Estimate
        
        R = bandit_arms[j].reward_fun()
        rewards[i] = R
        bandit_arms[j].update_fun(R)
    
    # Calculate Statistics!
    
    stats = {'Total Reward Earned': rewards.sum(), 'Overall Win Rate': rewards.sum()/T}
    stats.update({f'Steps Bandit Arm {bandit_arms.index(bandit_arm)}': bandit_arm.N for bandit_arm in bandit_arms})
    for key in stats:
        value = stats.get(key)
        print('\n', key, '\n\t', value, sep = '')
    
    win_rates = np.cumsum(rewards) / (np.arange(T) + 1)
    
    return win_rates

if __name__ == '__main__':
    
    q_vet = [0.25, 0.5, 0.75]
    T = 2000
    steps_draw = [0, 1, 5, 10, 50, 100, 250, 500, 1000, 1500, T-1]
    
    win_rates = experiment_fun(q_vet, T, steps_draw)
