################################
######## Inference Prod ########
################################

#%%# Catalyzer

# import re
import numpy as np
# import numba
# from scipy import interpolate
import torch
from sbi.inference import SNPE
from sbi import analysis
import time

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Inference Procedure (Prod) Class

class InferenceProd:
    
    def __init__(self, objective_fun, theta_set, prior, verbose = False):
        self.objective_fun = objective_fun
        self.theta_set = theta_set
        self.trajectory_set = None
        self.prior = prior
        self.posterior = None
        self.verbose = verbose
    
    def _preparator(self):
        self.objective_fun.verbose = False
        self.objective_fun.show = False
        self.objective_fun.apply()
        sap = self.objective_fun.data_objective.shape
        shape = (sap[0], sap[1]*sap[2]*sap[3])
        trajectory_set = torch.tensor(data = self.objective_fun.data_objective.reshape(shape), dtype = torch.float32)
        self.trajectory_set = trajectory_set
        return self
    
    def apply(self):
        self._preparator()
        inference = SNPE(self.prior)
        inference = inference.append_simulations(self.theta_set, self.trajectory_set)
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        self.posterior = posterior
        return self
    
    def _posterior_sampler(self, theta, observation, posterior_sample_shape, paras, para_span):
        posterior_samples = self.posterior.sample(sample_shape = posterior_sample_shape, x = observation)
        card = len(paras)
        if theta is None:
            synthetic = True
            points_colors = 'blue'
            theta = torch.quantile(posterior_samples, 0.5, 0) # torch.mean(posterior_samples, 0)
        else:
            synthetic = False
            points_colors = 'red'
        _ = analysis.pairplot(samples = posterior_samples, labels = paras, points = theta, points_colors = points_colors, points_offdiag = {'markersize': 4}, limits = [para_span]*card, figsize = (7.5, 7.5))
        plt.show()
        print(self.posterior)
        print(f'Theta\n\t{theta}')
        if synthetic:
            ret = (posterior_samples, theta)
        else:
            ret = posterior_samples
        return ret
    
    def examiner(self, trajectories, posterior_sample_shape, parameter_set_true):
        check = self.posterior is None
        mess = f'Please, we must execute/apply the inference procedure!posterior = {self.posterior}'
        assert not check, mess
        check = isinstance(trajectories, list)
        mess = 'We must provide a list of trajectory indexes!'
        assert check, mess
        cells = self.objective_fun.data.shape[3]
        x = np.arange(self.objective_fun.time_mini, self.objective_fun.time_maxi+1, self.objective_fun.time_delta)
        for trajectory in trajectories:
            for cell in range(cells):
                y = self.objective_fun.data[trajectory, :, :, cell].T
                plt.plot(x, y, drawstyle = 'steps-post', linestyle = '-')
                plt.title(f'Simulation ~ Cell\n{trajectory} ~ {cell}')
                plt.xlabel(self.objective_fun.time_unit)
                plt.ylabel('Copy Number')
                plt.legend(self.objective_fun.species)
                plt.grid(linestyle = '--')
                plt.show()
            observation = self.trajectory_set[trajectory]
            plt.plot(x, observation, drawstyle = 'steps-post', linestyle = '-')
            plt.title(f'Simulation\n{trajectory}')
            plt.xlabel(self.objective_fun.time_unit)
            plt.ylabel('Score')
            plt.xlim(self.objective_fun.time_mini, self.objective_fun.time_maxi)
            plt.ylim(-0.1, 1.1)
            plt.grid(linestyle = '--')
            plt.show()
        for trajectory in trajectories:
            theta = self.theta_set[trajectory]
            paras = list(parameter_set_true.keys())
            para_span = [0, 1]
            _ = self._posterior_sampler(theta, observation, posterior_sample_shape, paras, para_span)
        return None
    
    def synthesizer(self, observation, posterior_sample_shape, parameter_set_true, simulator_ready, objective_fun):
        check = self.posterior is None
        mess = f'Please, we must execute/apply the inference procedure!posterior = {self.posterior}'
        assert not check, mess
        check = isinstance(observation, torch.Tensor)
        mess = 'We must provide a valid PYTORCH tensor!'
        assert check, mess
        check = isinstance(objective_fun, type(self.objective_fun))
        mess = f"The 'objective_fun' object class must be consistent!\n\t'{type(objective_fun)} != {type(self.objective_fun)}'"
        assert check, mess
        check = objective_fun.data.ndim == 0
        mess = "We need an empty 'data' slot!\nValidation Condition!\n\tobjective_fun.data = None"
        assert check, mess
        theta = None
        paras = list(parameter_set_true.keys())
        para_span = [0, 1]
        _, theta = self._posterior_sampler(theta, observation, posterior_sample_shape, paras, para_span)
        data_synthetic = simulator_ready(theta.reshape(1, -1))
        objective_fun.data = data_synthetic
        objective_fun.verbose = False
        objective_fun.show = False
        objective_fun.apply()
        if self.verbose:
            objective_fun._previewer(objective_fun.data, objective_fun.species)
        x = np.arange(self.objective_fun.time_mini, self.objective_fun.time_maxi+1, self.objective_fun.time_delta)
        plt.plot(x, observation, drawstyle = 'steps-post', linestyle = '-')
        plt.plot(x, objective_fun.data_objective.reshape(-1, 1), drawstyle = 'steps-post', linestyle = '-')
        plt.title('Synthetic Comparison')
        plt.xlabel(self.objective_fun.time_unit)
        plt.ylabel('Score')
        plt.xlim(self.objective_fun.time_mini, self.objective_fun.time_maxi)
        plt.ylim(-0.1, 1.1)
        plt.legend(['User', 'Simulation'])
        plt.grid(linestyle = '--')
        plt.show()
        return None

#%%# Section [New]

