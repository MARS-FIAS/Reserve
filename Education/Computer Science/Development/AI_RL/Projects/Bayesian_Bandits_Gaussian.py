###########################################
######## Bayesian Bandits Gaussian ########
###########################################

#%%# Catalyzer

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
# import math

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Multi-Armed Bandits [Action-Value Methods] [Fixed Variance AND Random Mean]

class BanditArm: # Arm := Action # a\A
    
    def __init__(self, q = 0, tau = 1, seed = None):
        self.q = q # q = mu := True Mean # mu ~ N(mu_0, sigma_0^2) = N(Q, 1/tau_0) # q := True Action Value := Expected\Mean Reward | Action Selection
        self.tau = tau # tau := 1/sigma^2
        self.Q = 0 # Q = mu_0 := Estimated Mean # Q := Estimated Action Value # Mean [Prior]
        self.tau_0 = 1 # Precision [Prior]
        self.N = 0 # Action\Arm Selection Set Cardinality
        self.seed = seed
    
    def reward_fun(self): # x\R := Data # x|mu ~ N(mu, 1/tau)
        R = np.random.default_rng(seed = self.seed).normal(loc = self.q, scale = 1/np.sqrt(self.tau)) # Gaussian Distribution ~ N(q, 1/tau)
        return R
    
    def sample_fun(self):
        Q = np.random.default_rng(seed = self.seed).normal(loc = self.Q, scale = 1/np.sqrt(self.tau_0)) # Gaussian Distribution ~ N(Q, 1/tau_0)
        return Q
    
    def update_fun(self, R):
        self.N += 1
        self.Q = (self.tau * R + self.tau_0 * self.Q) / (self.tau_0 + self.tau) # Mean [Posterior]
        self.tau_0 += self.tau # Precision [Posterior]
        return self

def draw_fun(bandit_arms, t):
    x = np.linspace(-3, 5, 100*(3+5)+1) # q_vet
    for bandit_arm in bandit_arms:
        y = norm.pdf(x, bandit_arm.Q, 1/np.sqrt(bandit_arm.tau_0))
        plt.plot(x, y, label = f'q = {bandit_arm.q} | N = {bandit_arm.N}')
    plt.title(f'Bandit-Arm Posterior Distribution\nTime Steps = {t}')
    plt.legend()
    plt.grid(alpha = 0.25, linestyle = '--')
    plt.show()
    return None

def experiment_fun(q_vet, T, steps_draw, seed = None): # Time Steps := Action Selections
    
    bandit_arms = [BanditArm(q) for q in q_vet]
    rewards = np.zeros(T) # T := Final Time Step + 1 := Time Steps Number
    
    optimal_action = np.argmax([bandit_arm.q for bandit_arm in bandit_arms])
    print('Optimal Action:\t', optimal_action)
    
    for i in range(T): # i ~ t
        
        # Use Thompson Sampling
        
        j = np.argmax([bandit_arm.sample_fun() for bandit_arm in bandit_arms])
        
        # Plot Statistics!
        
        if i in steps_draw:
            draw_fun(bandit_arms, i)
        
        # Reward Function | Rewards Log | Update Action\Arm Value Estimate
        
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
    
    q_vet = [1, 2, 3]
    T = 2000
    steps_draw = [0, 1, 5, 10, 50, 100, 250, 500, 1000, 1500, T-1]
    
    win_rates = experiment_fun(q_vet, T, steps_draw)
    
    # Plot Statistics!
    
    cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
    plt.title('Bayesian Bandits Gaussian')
    plt.xlabel('Time Steps')
    plt.ylabel('Win Rates')
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid(alpha = 0.25, linestyle = '--')
    plt.hlines(y = q_vet, xmin = 0, xmax = T, colors = cocos[0:len(q_vet)], linestyles = '--')
    plt.plot(win_rates)
    plt.show()
