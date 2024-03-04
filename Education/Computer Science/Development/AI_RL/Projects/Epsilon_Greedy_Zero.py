#####################################
######## Epsilon-Greedy Zero ########
#####################################

#%%# Catalyzer

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Multi-Armed Bandits

trials = 10000
epsilon = 0.1 # (0, 1)
bandit_probabilities = [0.25, 0.5, 0.75] # True Win Probabilities

class Bandit:
    
    def __init__(self, p, seed = None):
        self.p = p # True Win Probability
        self.seed = seed
        self.p_estimate = 0
        self.N = 0 # Sample/Trial Collection Cardinality
    
    def pull(self):
        return np.random.default_rng(seed = self.seed).random() < self.p # Binomial Distribution
    
    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        return self
    
def experiment(seed = None):
    
    bandits = [Bandit(p) for p in bandit_probabilities]
    
    rewards = np.zeros(trials)
    trials_exploration = 0
    trials_exploitation = 0
    trials_optimal_action_selection = 0 # trials_optimal_bandit_selection
    
    optimal_action = np.argmax([bandit.p for bandit in bandits])
    print('Optimal Action:\t', optimal_action)
    
    for i in range(trials):
        
        # Use Epsilon-Greedy
        
        if np.random.default_rng(seed = seed).random() < epsilon:
            trials_exploration += 1
            j = np.random.default_rng(seed = seed).integers(len(bandits))
        else:
            trials_exploitation += 1
            j = np.argmax([bandit.p_estimate for bandit in bandits])
        
        if j == optimal_action:
            trials_optimal_action_selection += 1
        
        # Pull Arm | Update Rewards Log | Update Action/Bandit Value (Distribution)
        
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)
    
    for bandit in bandits:
        print('Mean Estimate:\t', bandit.p_estimate, '\t', bandits.index(bandit))
    
    # Calculate Statistics!
    
    stats = {'Total Reward Earned': rewards.sum(), 'Overall Win Rate': rewards.sum()/trials, 'Exploration Trials': trials_exploration, 'Exploitation Trials': trials_exploitation, 'Optimal Action/Bandit Selection Trials': trials_optimal_action_selection}
    for key in stats:
        value = stats.get(key)
        print('\n', key, '\n\t', value, sep = '')
    
    # Plot Statistics!
    
    win_rates = np.cumsum(rewards) / (np.arange(trials) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(trials)*np.max(bandit_probabilities))
    plt.show()
    
    return None

if __name__ == '__main__':
    experiment()
