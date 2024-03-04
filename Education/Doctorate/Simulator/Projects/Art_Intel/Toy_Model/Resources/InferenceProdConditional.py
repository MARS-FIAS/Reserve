############################################
######## Inference Prod Conditional ########
############################################

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

#%%# Inference Procedure (Prod) Conditional Class

class InferenceProd:
    
    def __init__(self, objective_fun, theta_set, prior, verbose = False):
        # Start
        self.objective_fun = objective_fun
        self.theta_set = theta_set
        self.trajectory_set = None
        self.prior = prior
        self.posterior = None
        self.verbose = verbose
        # Final
    
    def _preparator(self, keywords = dict()):
        self.objective_fun.verbose = False
        self.objective_fun.show = False
        tau_mini = keywords.get('tau_mini', None)
        tau_maxi = keywords.get('tau_maxi', None)
        tau_delta = keywords.get('tau_delta', None)
        self.objective_fun.apply(tau_mini = tau_mini, tau_maxi = tau_maxi, tau_delta = tau_delta)
        sap = self.objective_fun.data_objective.shape
        shape = (sap[0], sap[1]*sap[2]*sap[3])
        trajectory_set = torch.tensor(data = self.objective_fun.data_objective.reshape(shape), dtype = torch.float32)
        self.trajectory_set = trajectory_set
        return self
    
    def apply(self, **keywords):
        self._preparator(keywords = keywords)
        inference = SNPE(self.prior)
        inference = inference.append_simulations(self.theta_set, self.trajectory_set)
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        self.posterior = posterior
        return self
    
    def _posterior_sampler(self, theta, observation, posterior_sample_shape, paras, para_span, fig_size):
        posterior_samples = self.posterior.sample(sample_shape = posterior_sample_shape, x = observation)
        card = len(paras)
        if theta is None:
            synthetic = True
            points_colors = ['blue', 'magenta']
            theta_median = torch.quantile(posterior_samples, 0.5, 0) # torch.mean(posterior_samples, 0)
            self.posterior.set_default_x(observation) # MAP Estimation (Preparation)
            theta_mape = self.posterior.map(num_init_samples = posterior_sample_shape[0]) # MAP Estimation
            theta = {'median': theta_median, 'mape': theta_mape}
        else:
            synthetic = False
            points_colors = 'red'
        if self.verbose:
            minis = torch.floor(torch.min(posterior_samples, 0).values)
            maxis = torch.ceil(torch.max(posterior_samples, 0).values)
            check = torch.tensor([para_span[0] <= minis[index] and para_span[1] >= maxis[index] for index in range(posterior_samples.shape[1])])
            if torch.all(check):
                limes = [para_span.tolist()]*card
            else:
                limes = [[minis[index], maxis[index]] for index in range(posterior_samples.shape[1])]
            _ = analysis.pairplot(samples = posterior_samples, labels = paras, points = list(theta.values()) if synthetic else theta, points_colors = points_colors, points_offdiag = {'markersize': 4}, limits = limes, figsize = fig_size)
            plt.show()
        print(self.posterior)
        print(f'Theta\n\t{theta}')
        if synthetic:
            ret = (posterior_samples, theta)
        else:
            ret = posterior_samples
        return ret
    
    def examiner(self, trajectories, posterior_sample_shape, parameter_set_true, **keywords):
        check = self.posterior is None
        mess = f'Please, we must execute/apply the inference procedure!\nposterior = {self.posterior}'
        assert not check, mess
        check = isinstance(trajectories, list)
        mess = 'We must provide a list of trajectory indexes!'
        assert check, mess
        cells = self.objective_fun.data.shape[3]
        x = self.objective_fun.tau # np.arange(self.objective_fun.time_mini, self.objective_fun.time_maxi+1, self.objective_fun.time_delta)
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
            para_span = torch.tensor([0, 1])
            _ = self._posterior_sampler(theta, observation, posterior_sample_shape, paras, para_span, fig_size = keywords.get('fig_size', (7.5, 7.5)))
        return None
    
    def synthesizer(self, observation, posterior_sample_shape, parameter_set_true, simulator_ready, objective_fun, **keywords):
        check = self.posterior is None
        mess = f'Please, we must execute/apply the inference procedure!\nposterior = {self.posterior}'
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
        para_span = torch.tensor([0, 1])
        posterior_samples, theta = self._posterior_sampler(theta, observation, posterior_sample_shape, paras, para_span, fig_size = keywords.get('fig_size', (7.5, 7.5)))
        self.synth_posterior_samples = posterior_samples # Important Info!
        self.synth_theta = theta # Important Info!
        data_synthetic = simulator_ready(theta['mape'].reshape(1, -1))
        objective_fun.data = data_synthetic
        objective_fun.verbose = False
        objective_fun.show = False
        objective_fun.apply(tau_mini = self.objective_fun.time_mini, tau_maxi = self.objective_fun.time_maxi, tau_delta = self.objective_fun.time_delta)
        if self.verbose:
            objective_fun._previewer(objective_fun.data, objective_fun.species)
        if self.verbose:
            x = self.objective_fun.tau # np.arange(self.objective_fun.time_mini, self.objective_fun.time_maxi+1, self.objective_fun.time_delta)
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

