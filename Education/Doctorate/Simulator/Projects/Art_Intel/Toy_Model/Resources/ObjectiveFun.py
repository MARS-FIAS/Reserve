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
        self.tau = None
        self.simulations_maxi = simulations_maxi
        self.verbose = verbose
        self.show = show
    
    def _check_data_shape(self):
        shape = ('simulations', 'len(species)', 'len(x)', 'cells')
        check = len(self.data.shape) == len(shape)
        mess = f'Please, we must restructure/reshape the data!\n\tshape = {shape}'
        assert check, mess
        return None
    
    def _preview_data(self, data, species):
        self._check_data_shape() # Inspection!
        simulations = data.shape[0]
        cells = data.shape[3] # data.shape[-1]
        if self.tau is None:
            x = np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
        else:
            x = self.tau
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
    
    def restructure_data(self):
        noms = {'Seconds': 0, 'Minutes': 1, 'Hours': 2}
        _nom = noms[self.time_unit]
        nom = np.power(60, _nom)
        xl = self.time_mini*nom
        xr = self.time_maxi*nom
        x = np.arange(xl, xr+nom, self.time_delta*nom)/nom
        self.tau = x
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
            self._preview_data(self.data_objective, self.species_objective)
        return self
    
    def decimate_data(self, tau_mini = None, tau_maxi = None, tau_delta = None):
        mess = f"We must be consistent!\n\tTime Unit: '{self.time_unit}'!"
        if self.verbose: print(mess) # Descript!
        if (tau_mini, tau_maxi) == (None, None):
            pass
        else:
            noms = {'Seconds': 0, 'Minutes': 1, 'Hours': 2}
            _nom = noms[self.time_unit]
            nom = np.power(60, _nom)
            xl = self.time_mini*nom
            xr = self.time_maxi*nom
            x = np.arange(xl, xr+nom, self.time_delta*nom)/nom
            if not(tau_delta is None):
                _tau_delta = int(tau_delta*nom)
                _time_delta = int(self.time_delta*nom)
                check_alp = _tau_delta // _time_delta
                check_bet = _tau_delta % _time_delta
                check = check_alp >= 1 and check_bet == 0
                mess = f"Oops! Impossible! The number '{tau_delta}' must be a positive integer multiple of the number '{self.time_delta}'!"
                assert check, mess
            _tau = np.array([tau_mini, tau_maxi, tau_delta])
            test = np.array([tau is None for tau in _tau])
            patch = np.array([np.min(x), np.max(x), self.time_delta])
            _tau[test] = patch[test]
            test = np.array([not(tau in x) for tau in _tau[0:2]])
            patch = np.array([x[np.argmin(np.abs(x-_tau[0]))], x[np.argmin(np.abs(x-_tau[1]))]])
            _tau[0:2][test] = patch[test]
            paces = int((_tau[1]-_tau[0])/_tau[2])+1
            tau = np.array([_tau[0]+index*_tau[2] for index in range(paces)])
            _tau[0:2] = np.array([np.min(tau), np.max(tau)])
            where = np.array([np.argmin(np.abs(x-index)) for index in tau])
            check = where.shape == tau.shape
            mess = "Oops! Something went wrong! Our algorithm has a bug!\nWe must check the 'tau' construction!"
            assert check, mess
            self.data = self.data[:, :, where, :]
            self.data_objective = self.data_objective[:, :, where, :]
            self.time_mini = _tau[0]
            self.time_maxi = _tau[1]
            self.time_delta = _tau[2]
            self.tau = tau
            if self.verbose:
                print('Decimate Data!\n\t', self.data_objective.shape, '\n\t', self.species_objective, '\n\t', _tau, sep = '')
                self._preview_data(self.data_objective, self.species_objective)
        return self
    
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
            if self.show: self._preview_data(self.data, self.species)
            print('Sieve Data!\n\t', self.data_objective.shape, '\n\t', self.species_objective, sep = '')
            self._preview_data(self.data_objective, self.species_objective)
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
            if self.show: self._preview_data(self.data, self.species)
            print('Comb Tram Data!\n\t', self.data_objective.shape, '\n\t', self.species_objective, sep = '')
            self._preview_data(self.data_objective, self.species_objective)
        return self
    
    def validate_data(self):
        pass
    
    def _visualize_data_preprocessor(self, data_preprocessor):
        pass
    
    def visualize_data_objective(self):
        pass
    
    def apply(self, **keywords):
        pass
    
    def appraise(self, **keywords):
        pass

#%%# Objective Fun Class [Zero]

class ObjectiveFunZero(ObjectiveFunTemplate):
    
    def __init__(self, data, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, verbose = False, show = False):
        super().__init__(data, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, verbose, show)
    
    def apply(self):
        # Data Preprocessor! [Start]
        self.restructure_data() # Step Zero!
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
        x = self.tau # np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
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
    
    def __init__(self, data, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, verbose = False, show = False, **keywords):
        super().__init__(data, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, verbose, show)
        self.alpha = keywords.get('alpha', 0)
        self.maxi_G = keywords.get('maxi_G', 1000)
        self.maxi_NT = keywords.get('maxi_NT', 500)
    
    def apply(self, **keywords):
        # Data Preprocessor! [Start]
        self.restructure_data() # Step Zero!
        tau_mini = keywords.get('tau_mini', None)
        tau_maxi = keywords.get('tau_maxi', None)
        tau_delta = keywords.get('tau_delta', None)
        self.decimate_data(tau_mini, tau_maxi, tau_delta)
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
        x = self.tau # np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
        D = list(range(4))
        U = list(range(4, cells-1))
        NT_D = np.sum(self.data_objective[:, self.species_objective.index('NT'), :, D], 0) # NT Down
        NT_U = np.sum(self.data_objective[:, self.species_objective.index('NT'), :, U], 0) # NT Up
        G_D = np.sum(self.data_objective[:, self.species_objective.index('G'), :, D], 0) # G Down
        G_U = np.sum(self.data_objective[:, self.species_objective.index('G'), :, U], 0) # G Up
        p = np.power(np.minimum(1, G_D/(len(D)*self.maxi_G)), self.alpha)
        q = np.power(np.minimum(1, NT_U/(len(U)*self.maxi_NT)), self.alpha)
        l = G_D/(NT_D+G_D) # G_D/(np.max(NT_D)+np.max(G_D))
        r = NT_U/(NT_U+G_U) # NT_U/(np.max(NT_U)+np.max(G_U))
        data_objective[:, 0, :, 0] = 0.5*(p*l+q*r)
        check = np.isnan(data_objective)
        if np.any(check):
            data_objective[check] = 0 # Default Value = 0
            mess = '~¡!~'*8+"\n\tSeveral time series have 'nans'!\n\tWe have replaced those nan values with 'zeros'!\n"+'~¡!~'*8
            print(mess)
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
        if self.verbose or self.show:
            for simulation in range(simulations):
                if simulation >= self.simulations_maxi:
                    break
                plt.plot(x, self.data_objective[simulation, 0, :, 0], color = 'tab:purple')
                plt.title(f'Simulation\n{simulation}\nAlpha\n{self.alpha}')
                plt.xlabel(self.time_unit)
                plt.ylabel('Score')
                plt.xlim(self.time_mini, self.time_maxi)
                plt.ylim(-0.1, 1.1)
                plt.grid(linestyle = '--')
                plt.show()
        return self
    
    def appraise(self, **keywords):
        check = self.data_objective is None
        mess = f'Please, we must synthesize (or execute/apply) the inference procedure!\ndata_objective = {self.data_objective}'
        assert not check, mess
        perks = keywords.get('perks', [10, 50, 90])
        phi = keywords.get('phi', 0.5**2)
        psi = keywords.get('psi', 1 - phi)
        check = phi + psi == 1
        mess = f"Oops! The sum of the two numbers 'phi' and 'psi' does not equal '1'!\n\t'phi + psi = {phi + psi}'"
        assert check, mess
        percentiles = np.percentile(a = self.data_objective, q = perks, axis = 0).reshape((len(perks), -1)).T
        alp = 1 - (percentiles[:, 2] - percentiles[:, 0])
        bet = percentiles[:, 1]
        chi = phi * alp + psi * bet
        appraisal = np.mean(chi)
        if self.verbose or self.show:
            cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
            x = self.tau # np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
            x_mini, x_maxi = np.min(self.tau), np.max(self.tau)
            simulations = self.data_objective.shape[0]
            for simulation in range(simulations):
                if simulation >= self.simulations_maxi:
                    break
                coco = cocos[simulation % len(cocos)]
                plt.plot(x, self.data_objective[simulation, 0, :, 0], color = coco, alpha = 0.25)
                plt.title(f"Train Trial\n{keywords.get('train_trial', 0)}\nAppraisal\n{appraisal}")
                plt.xlabel(self.time_unit)
                plt.ylabel('Appraisal')
                plt.xlim(x_mini, x_maxi)
                plt.ylim(0, 1)
                plt.grid(linestyle = '--', alpha = 0.25)
            plt.plot(x, percentiles[:, 0], linestyle = '--', color = 'r')
            plt.plot(x, percentiles[:, 1], linestyle = '--', color = 'g')
            plt.plot(x, percentiles[:, 2], linestyle = '--', color = 'b')
            plt.plot(x, chi, linestyle = '-', color = 'm')
            plt.hlines(appraisal, x_mini, x_maxi, colors = 'k', alpha = 0.5)
            plt.show()
        return appraisal

#%%# Section [New]

