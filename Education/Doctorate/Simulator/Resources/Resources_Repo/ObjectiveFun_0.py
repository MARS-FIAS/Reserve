###############################
######## Objective Fun ########
###############################

#%%# Catalyzer

# import re
import numpy as np
# import numba
# from scipy import interpolate
# import torch
# from sbi.inference import simulate_for_sbi
# import time

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Objective Fun Class [Template]

class ObjectiveFunTemplate:
    
    def __init__(self, data, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, verbose = False, show = False):
        self.data = np.copy(data) # Welfare purpose!
        self.data_objective = None
        self.species = species
        self.species_objective = None
        self.time_mini = time_mini
        self.time_maxi = time_maxi
        self.time_unit = time_unit
        self.time_delta = time_delta
        self.simulations_maxi = simulations_maxi
        self.verbose = verbose
        self.show = show
    
    def _check_data_shape(self):
        shape = ('simulations', 'len(species)', 'len(x)', 'cells')
        check = len(self.data.shape) == len(shape)
        mess = f'Please, we must restructure/reshape the data!\n\tshape = {shape}'
        assert check, mess
        return None
    
    def _previewer(self, data, species):
        self._check_data_shape() # Inspection!
        simulations = data.shape[0]
        cells = data.shape[3] # data.shape[-1]
        x = np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
        for simulation in range(simulations):
            if simulation >= self.simulations_maxi:
                break
            for cell in range(cells):
                y = data[simulation, :, :, cell].T
                plt.plot(x, y, drawstyle = 'steps-post', linestyle = '-')
                plt.title(f'Simulation ~ Cell\n{simulation} ~ {cell}')
                plt.xlabel(self.time_unit)
                plt.ylabel('Copy Number')
                plt.legend(species)
                plt.grid(linestyle = '--')
                plt.show()
        return None
    
    def restructure(self):
        noms = {'Seconds': 0, 'Minutes': 1, 'Hours': 2}
        _nom = noms[self.time_unit]
        nom = np.power(60, _nom)
        xl = self.time_mini
        xr = self.time_maxi*nom
        x = np.arange(xl, xr+nom, self.time_delta*nom)/nom
        if len(self.data.shape) == 1:
            simulations = 1
            take = 0
        else: # len(self.data.shape) == 2
            simulations = self.data.shape[0]
            take = 1
        _cells = self.data.shape[take]/len(self.species)
        cells = int(_cells/len(x))
        data_rest = self.data.reshape((simulations, len(self.species), len(x), cells))
        self.data = data_rest
        self.data_objective = data_rest
        self.species_objective = self.species.copy()
        if self.verbose:
            print('Restructure Data!\n\t', self.data_objective.shape, '\n\t', self.species_objective, sep = '')
            self._previewer(self.data_objective, self.species_objective)
        return self
    
    def decimator(self):
        pass
    
    def sieve_data(self, species_sieve):
        mess = "species_sieve = ['old_species_0', ..., 'old_species_10']"
        if self.verbose: print(mess) # Descript!
        self._check_data_shape() # Inspection!
        check = [s in self.species_objective for s in species_sieve]
        mess = f"The list 'species_sieve' is not consistent with the 'species_objective' list!\n\t{np.array(species_sieve)[np.invert(check)]}"
        assert all(check), mess
        indices = [self.species_objective.index(s) for s in species_sieve]
        data_sieve = self.data_objective[:, indices, :, :]
        self.data_objective = data_sieve
        self.species_objective = species_sieve
        if self.verbose:
            if self.show: self._previewer(self.data, self.species)
            print('Sieve Data!\n\t', self.data_objective.shape, '\n\t', self.species_objective, sep = '')
            self._previewer(self.data_objective, self.species_objective)
        return self
    
    def comb_tram_data(self, species_comb_tram_dit):
        mess = "species_comb_tram_dit = {'new_species': (('old_species_0', argument_0, ..., 'old_species_10', argument_10), comb_tram_function)}"
        if self.verbose: print(mess) # Descript!
        self._check_data_shape() # Inspection!
        specs = list(species_comb_tram_dit.keys())
        check = [spec in self.species_objective for spec in specs]
        mess = f'We must provide a new name for each novel species!\n\t{specs}'
        assert not any(check), mess
        species_comb_tram = self.species_objective.copy()
        species_comb_tram.extend(specs)
        data_comb_tram = self.data_objective
        for spec in specs:
            elements = species_comb_tram_dit.get(spec)
            spas = elements[0]
            check_alp = [isinstance(spa, str) for spa in spas]
            mess = 'At least one of the arguments must be a valid species name or string!'
            assert any(check_alp), mess
            check_bet = np.array(spas)[check_alp]
            check = [s in self.species_objective for s in check_bet]
            mess = f'Invalid species!\n\t{check_bet}'
            assert all(check), mess
            fun = elements[1]
            check = isinstance(fun, np.ufunc)
            mess = "The 'comb_tram' function must be a 'NUMPY' universal function"
            assert check, mess
            arguments = [data_comb_tram[:, [self.species_objective.index(spa)], :, :] if spa in self.species_objective else spa for spa in spas]
            temp = fun(*arguments)
            data_comb_tram = np.append(data_comb_tram, temp, 1)
        self.data_objective = data_comb_tram
        self.species_objective = species_comb_tram
        if self.verbose:
            if self.show: self._previewer(self.data, self.species)
            print('Comb Tram Data!\n\t', self.data_objective.shape, '\n\t', self.species_objective, sep = '')
            self._previewer(self.data_objective, self.species_objective)
        return self
    
    def apply(self):
        pass
    
    def visualizer(self):
        pass

#%%# Objective Fun Class [Zero]

class ObjectiveFunZero(ObjectiveFunTemplate):
    
    def __init__(self, data, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, verbose = False, show = False):
        super().__init__(data, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, verbose, show)
    
    def apply(self):
        # Data Preprocessor! [Start]
        self.restructure() # Step Zero!
        species_sieve = ['N', 'G', 'NP']
        self.sieve_data(species_sieve)
        species_comb_tram_dit = {'NT': (('N', 'NP'), np.add)} # {'GP': (('G', 0.5), np.power)}
        self.comb_tram_data(species_comb_tram_dit)
        species_sieve = ['NT', 'G']
        self.sieve_data(species_sieve)
        data_preprocessor = self.data_objective # Temp Data!
        # Data Preprocessor! [Final]
        # Data Processor! [Start]
        data_objective = np.full((self.data_objective.shape[0], 1, self.data_objective.shape[2], self.data_objective.shape[3]), np.nan)
        simulations = self.data_objective.shape[0]
        cells = self.data_objective.shape[-1] # data.shape[3]
        x = np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
        NT_E = np.sum(self.data_objective[:, self.species_objective.index('NT'), :, :], 2) # Embryo
        NT_C = np.copy(self.data_objective[:, self.species_objective.index('NT'), :, :]) # Cell
        G_E = np.sum(self.data_objective[:, self.species_objective.index('G'), :, :], 2) # Embryo
        G_C = np.copy(self.data_objective[:, self.species_objective.index('G'), :, :]) # Cell
        a = NT_C/(NT_C+G_C)
        b = G_C/(NT_C+G_C)
        c = NT_E/(NT_E+G_E)
        d = G_E/(NT_E+G_E)
        for cell in range(4):
            data_objective[:, 0, :, cell] = d*b[..., cell]/(1-(d*a[..., cell]+c*b[..., cell]))
        for cell in range(4, cells-1):
            data_objective[:, 0, :, cell] = c*a[..., cell]/(1-(d*a[..., cell]+c*b[..., cell]))
        self.data_objective = data_objective
        # Data Processor! [Final]
        if self.verbose:
            for simulation in range(simulations):
                if simulation >= self.simulations_maxi:
                    break
                for cell in range(cells-1):
                    tit = f'Simulation ~ Cell\n{simulation} ~ {cell}'
                    y = data_preprocessor[simulation, :, :, cell].T
                    plt.plot(x, y, drawstyle = 'steps-post', linestyle = '-')
                    plt.title(tit)
                    plt.xlabel(self.time_unit)
                    plt.ylabel('Copy Number')
                    plt.legend(self.species_objective)
                    plt.grid(linestyle = '--')
                    plt.show()
                    color = 'tab:green' if cell >= 4 else 'tab:red'
                    plt.plot(x, self.data_objective[simulation, 0, :, cell], color = color)
                    plt.title(tit)
                    plt.xlabel(self.time_unit)
                    plt.ylabel('Score')
                    plt.xlim(self.time_mini, self.time_maxi)
                    plt.ylim(-0.1, 1.1)
                    plt.grid(linestyle = '--')
                    plt.show()
        return self

#%%# Objective Fun Class [One]

class ObjectiveFunOne(ObjectiveFunTemplate):
    
    def __init__(self, data, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, verbose = False, show = False):
        super().__init__(data, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, verbose, show)
    
    def apply(self):
        # Data Preprocessor! [Start]
        self.restructure() # Step Zero!
        species_sieve = ['N', 'G', 'NP']
        self.sieve_data(species_sieve)
        species_comb_tram_dit = {'NT': (('N', 'NP'), np.add)} # {'GP': (('G', 0.5), np.power)}
        self.comb_tram_data(species_comb_tram_dit)
        species_sieve = ['NT', 'G']
        self.sieve_data(species_sieve)
        data_preprocessor = self.data_objective # Temp Data!
        # Data Preprocessor! [Final]
        # Data Processor! [Start]
        data_objective = np.full((self.data_objective.shape[0], 1, self.data_objective.shape[2], 1), np.nan)
        simulations = self.data_objective.shape[0]
        cells = self.data_objective.shape[-1] # data.shape[3]
        x = np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
        D = list(range(4))
        U = list(range(4, cells-1))
        NT_D = np.sum(self.data_objective[:, self.species_objective.index('NT'), :, D], 0) # NT Down
        NT_U = np.sum(self.data_objective[:, self.species_objective.index('NT'), :, U], 0) # NT Up
        G_D = np.sum(self.data_objective[:, self.species_objective.index('G'), :, D], 0) # G Down
        G_U = np.sum(self.data_objective[:, self.species_objective.index('G'), :, U], 0) # G Up
        l = G_D/(NT_D+G_D) # G_D/(np.max(NT_D)+np.max(G_D))
        r = NT_U/(NT_U+G_U) # NT_U/(np.max(NT_U)+np.max(G_U))
        data_objective[:, 0, :, 0] = 0.5*(l+r)
        self.data_objective = data_objective
        # Data Processor! [Final]
        if self.verbose:
            for simulation in range(simulations):
                if simulation >= self.simulations_maxi:
                    break
                for cell in range(cells-1):
                    y = data_preprocessor[simulation, :, :, cell].T
                    plt.plot(x, y, drawstyle = 'steps-post', linestyle = '-')
                    plt.title(f'Simulation ~ Cell\n{simulation} ~ {cell}')
                    plt.xlabel(self.time_unit)
                    plt.ylabel('Copy Number')
                    plt.legend(self.species_objective)
                    plt.grid(linestyle = '--')
                    plt.show()
                plt.plot(x, self.data_objective[simulation, 0, :, 0], color = 'tab:purple')
                plt.title(f'Simulation\n{simulation}')
                plt.xlabel(self.time_unit)
                plt.ylabel('Score')
                plt.xlim(self.time_mini, self.time_maxi)
                plt.ylim(-0.1, 1.1)
                plt.grid(linestyle = '--')
                plt.show()
        return self

#%%# Section [New]

