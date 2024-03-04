####################################
######## Epsilon-Greedy One ########
####################################

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
    
    def __init__(self, action_value_true, seed = None):
        self.action_value_true = action_value_true # q := True Action Value := Expected/Mean Reward | Action Selection
        self.action_value_estimate = 0 # Q := Estimated Action Value
        self.N = 0 # Action/Arm Selection Set Cardinality
        self.seed = seed
    
    def reward_fun(self):
        R = np.random.default_rng(seed = self.seed).normal(loc = self.action_value_true, scale = 1) # Gaussian Distribution # R = np.random.default_rng(seed = self.seed).random() < self.action_value_true # Binomial Distribution
        return R
    
    def update_fun(self, R):
        self.N += 1
        self.action_value_estimate = ((self.N - 1) * self.action_value_estimate + R) / self.N
        return self

def experiment_fun(epsilon, action_value_true_vet, T, seed = None): # Time Steps := Action Selections
    
    bandit_arms = [BanditArm(q) for q in action_value_true_vet]
    
    rewards = np.zeros(T) # T := Final Time Step + 1 := Time Steps Number
    steps_exploration = 0
    steps_exploitation = 0
    steps_optimal_action_selection = 0 # steps_optimal_arm_selection
    
    optimal_action = np.argmax([bandit_arm.action_value_true for bandit_arm in bandit_arms])
    print('Optimal Action:\t', optimal_action)
    
    for i in range(T): # i ~ t
        
        # Use Epsilon-Greedy
        
        if np.random.default_rng(seed = seed).random() < epsilon:
            steps_exploration += 1
            j = np.random.default_rng(seed = seed).integers(len(bandit_arms))
        else:
            steps_exploitation += 1
            j = np.argmax([bandit_arm.action_value_estimate for bandit_arm in bandit_arms])
        
        if j == optimal_action: # j ~ A
            steps_optimal_action_selection += 1
        
        # Reward Function | Rewards Log | Update Action/Arm Value Estimate
        
        R = bandit_arms[j].reward_fun()
        rewards[i] = R
        bandit_arms[j].update_fun(R)
    
    for bandit_arm in bandit_arms:
        print('Action Value Estimate:\t', bandit_arm.action_value_estimate, '\t', bandit_arms.index(bandit_arm))
    
    # Calculate Statistics!
    
    stats = {'Total Reward Earned': rewards.sum(), 'Overall Win Rate': rewards.sum()/T, 'Exploration Steps': steps_exploration, 'Exploitation Steps': steps_exploitation, 'Optimal Action/Arm Selection Steps': steps_optimal_action_selection}
    for key in stats:
        value = stats.get(key)
        print('\n', key, '\n\t', value, sep = '')
    
    # Plot Statistics!
    
    win_rates = np.cumsum(rewards) / (np.arange(T) + 1)
    
    cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
    plt.title('Epsilon = ' + str(epsilon))
    plt.xlabel('Time Steps')
    plt.ylabel('Win Rates')
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlim(1, T)
    plt.ylim(np.min(action_value_true_vet)-np.min(action_value_true_vet)/10, np.max(action_value_true_vet)+np.max(action_value_true_vet)/10)
    plt.grid(alpha = 0.25, linestyle = '--')
    plt.hlines(y = action_value_true_vet, xmin = 0, xmax = T, colors = cocos[0:len(action_value_true_vet)], linestyles = '--')
    plt.plot(win_rates, color = 'tab:gray')
    plt.show()
    
    return win_rates

if __name__ == '__main__':
    
    data = []
    epsilons = [0.01, 0.05, .1]
    action_value_true_vet = [2.5, 5, 7.5]
    T = 10000
    
    for epsilon in epsilons:
        win_rates = experiment_fun(epsilon, action_value_true_vet, T)
        data.append(win_rates)
    
    plt.title('Epsilon Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Win Rates')
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlim(1, T)
    plt.ylim(np.min(action_value_true_vet)-np.min(action_value_true_vet)/10, np.max(action_value_true_vet)+np.max(action_value_true_vet)/10)
    plt.grid(alpha = 0.25, linestyle = '--')
    plt.hlines(y = action_value_true_vet, xmin = 0, xmax = T, colors = 'tab:gray', linestyles = '--')
    for datum in data:
        plt.plot(datum)
    plt.legend([None]+epsilons)
    plt.show()
