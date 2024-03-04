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
        self.data_rule = None
        self.data_objective = None
        self.species = species
        self.species_objective = None
        self.time_mini = time_mini
        self.time_maxi = time_maxi
        self.time_unit = time_unit
        self.time_delta = time_delta
        self._time_mini = time_mini # Welfare purpose!
        self._time_maxi = time_maxi # Welfare purpose!
        self._time_unit = time_unit # Welfare purpose!
        self._time_delta = time_delta # Welfare purpose!
        self.tau = None
        self.simulations_maxi = simulations_maxi
        self.verbose = verbose
        self.show = show
    
    def _check_data_shape(self, data = None):
        if data is None:
            data = self.data
        shape_alp = ('simulations', 'len(species)', 'len(x)', 'cells')
        shape_bet = ('len(rules)', *shape_alp)
        check = len(data.shape) == len(shape_alp) or len(data.shape) == len(shape_bet)
        mess = f'Please, we must restructure/reshape the data!\n\tshape = {shape_alp} OR shape = {shape_bet}'
        assert check, mess
        return None
    
    def _preview_data(self, data, species):
        self._check_data_shape(data) # Inspection!
        if len(data.shape) == 4: # shape_alp = ('simulations', 'len(species)', 'len(x)', 'cells')
            rule_indices = [0]
            take = 0
        else: # len(data.shape) == 5 # shape_bet = ('len(rules)', *shape_alp)
            rule_indices = range(data.shape[0])
            take = 1
        simulations = data.shape[take]
        cells = data.shape[3+take] # data.shape[-1]
        if self.tau is None:
            x = np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
        else:
            x = self.tau
        for rule_index in rule_indices:
            if take:
                data_preview = data[rule_index]
            else:
                data_preview = data
            for simulation in range(simulations):
                if simulation >= self.simulations_maxi:
                    break
                for cell in range(cells):
                    y = data_preview[simulation, :, :, cell].T
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
            self.tau_where = np.arange(self.tau.size)
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
            self.tau_where = where # Metamorphose Data!
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

#%%# Objective Fun Class [Portion]

class ObjectiveFunPortion(ObjectiveFunTemplate):
    
    def __init__(self, data, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, verbose = False, show = False, **keywords):
        super().__init__(data, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, verbose, show)
        self.beta = keywords.get('beta', 1)
        self.positive_NT = keywords.get('positive_NT', 500)
        self.negative_NT = keywords.get('negative_NT', 200)
        self.positive_G = keywords.get('positive_G', 1000)
        self.negative_G = keywords.get('negative_G', 200)
        self.cusp_NT = keywords.get('cusp_NT', 5)
        self.cusp_G = keywords.get('cusp_G', 5)
        self.aim_NT = keywords.get('aim_NT', 4/10)
        self.aim_G = keywords.get('aim_G', 1-self.aim_NT)
    
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
        cells = self.data_objective.shape[-1] # self.data.shape[3]
        x = self.tau # np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
        threshold_NT_positive = self.positive_NT-self.cusp_NT*np.sqrt(self.positive_NT)
        threshold_NT_negative = self.negative_NT+self.cusp_NT*np.sqrt(self.negative_NT)
        threshold_G_positive = self.positive_G-self.cusp_G*np.sqrt(self.positive_G)
        threshold_G_negative = self.negative_G+self.cusp_G*np.sqrt(self.negative_G)
        NT_positive = self.data_objective[:, self.species_objective.index('NT'), :, :] > threshold_NT_positive
        NT_negative = self.data_objective[:, self.species_objective.index('NT'), :, :] < threshold_NT_negative
        G_positive = self.data_objective[:, self.species_objective.index('G'), :, :] > threshold_G_positive
        G_negative = self.data_objective[:, self.species_objective.index('G'), :, :] < threshold_G_negative
        classification = { # (NT, G) # '(+|-)(+|-)'
            '++': np.logical_and(NT_positive, G_positive),
            '+-': np.logical_and(NT_positive, G_negative),
            '-+': np.logical_and(NT_negative, G_positive),
            '--': np.logical_and(NT_negative, G_negative)
        }
        _NT_portion = np.count_nonzero(classification['+-'], 2)
        _G_portion = np.count_nonzero(classification['-+'], 2)
        NT_portion = _NT_portion/cells
        G_portion = _G_portion/cells
        NT = np.abs(self.aim_NT-NT_portion)
        G = np.abs(self.aim_G-G_portion)
        delta = np.exp(-self.beta*2*np.maximum(self.aim_NT, self.aim_G))
        phi = 1-delta
        score = (np.exp(-self.beta*(NT+G))-delta)/phi
        data_objective[:, 0, :, 0] = score
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
                plt.title(f'Simulation\n{simulation}\nBeta\n{self.beta}')
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

#%%# Objective Fun Class [Portion Rule]

class ObjectiveFunPortionRule(ObjectiveFunTemplate):
    
    def __init__(self, data, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, verbose = False, show = False, **keywords):
        super().__init__(data, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, verbose, show)
        self.rules = keywords.get('rules', [0, 1])
        self.beta = keywords.get('beta', 1)
        self.positive_NT = keywords.get('positive_NT', 500)
        self.negative_NT = keywords.get('negative_NT', 200)
        self.positive_G = keywords.get('positive_G', 1000)
        self.negative_G = keywords.get('negative_G', 200)
        self.cusp_NT = keywords.get('cusp_NT', 5)
        self.cusp_G = keywords.get('cusp_G', 5)
        self.aim_NT = keywords.get('aim_NT', [1, 4/10])
        self.aim_G = keywords.get('aim_G', [1-self.aim_NT[0], 1-self.aim_NT[1]])
    
    def prepare_data_rule(self):
        if len(self.data.shape) == 1:
            self.data_rule = np.copy(self.data.reshape((1, self.data.shape[0]))) # Welfare purpose!
        else:
            self.data_rule = np.copy(self.data) # Welfare purpose!
        return self
    
    def metamorphose_data(self):
        check_alp = [self.data_rule, self.data_objective, self.species_objective, self.tau, self.tau_where]
        check_bet = [_ is None for _ in check_alp]
        check = not any(check_bet)
        mess = "Oops! Something went wrong! We must execute/apply the 'objective function' process!"
        assert check, mess
        self.data = self.data_rule
        simulations = self.data.shape[0]
        x = np.arange(self._time_mini, self._time_maxi+1, self._time_delta)
        cells_alp = self.data.shape[1]/len(self.rules)
        cells_bet = cells_alp/len(self.species)
        cells = int(cells_bet/len(x))
        data_rest = self.data.reshape((len(self.rules), simulations, len(self.species), len(x), cells))
        self.data = data_rest[:, :, :, self.tau_where, :]
        if self.verbose:
            print("Metamorphose Data! 'Data Rule!'\n\t", self.data.shape, '\n\t', self.species, sep = '')
            self._preview_data(self.data, self.species)
        return self
    
    def apply(self, **keywords):
        # Data Preprocessor! [Start]
        self.prepare_data_rule()
        splitter = int(self.data_rule.shape[1]/len(self.rules)) # Slice
        rule_split = [(self.rules.index(rule)*splitter, (self.rules.index(rule)+1)*splitter) for rule in self.rules]
        for rule in self.rules:
            rule_index = self.rules.index(rule)
            if self.verbose: print(f"{'~'*8} Rule! {rule_index+1} : {len(self.rules)} {'~'*8}")
            split = rule_split[rule_index]
            self.data = self.data_rule[:, split[0]:split[1]]
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
            cells = self.data_objective.shape[-1] # self.data.shape[3]
            x = self.tau # np.arange(self.time_mini, self.time_maxi+1, self.time_delta)
            threshold_NT_positive = self.positive_NT-self.cusp_NT*np.sqrt(self.positive_NT)
            threshold_NT_negative = self.negative_NT+self.cusp_NT*np.sqrt(self.negative_NT)
            threshold_G_positive = self.positive_G-self.cusp_G*np.sqrt(self.positive_G)
            threshold_G_negative = self.negative_G+self.cusp_G*np.sqrt(self.negative_G)
            NT_positive = self.data_objective[:, self.species_objective.index('NT'), :, :] > threshold_NT_positive
            NT_negative = self.data_objective[:, self.species_objective.index('NT'), :, :] < threshold_NT_negative
            G_positive = self.data_objective[:, self.species_objective.index('G'), :, :] > threshold_G_positive
            G_negative = self.data_objective[:, self.species_objective.index('G'), :, :] < threshold_G_negative
            classification = { # (NT, G) # '(+|-)(+|-)'
                '++': np.logical_and(NT_positive, G_positive),
                '+-': np.logical_and(NT_positive, G_negative),
                '-+': np.logical_and(NT_negative, G_positive),
                '--': np.logical_and(NT_negative, G_negative)
            }
            _NT_portion = np.count_nonzero(classification['+-'], 2)
            _G_portion = np.count_nonzero(classification['-+'], 2)
            NT_portion = _NT_portion/cells
            G_portion = _G_portion/cells
            NT = np.abs(self.aim_NT[rule_index]-NT_portion)
            G = np.abs(self.aim_G[rule_index]-G_portion)
            delta = np.exp(-self.beta*2*np.maximum(self.aim_NT[rule_index], self.aim_G[rule_index]))
            phi = 1-delta
            score = (np.exp(-self.beta*(NT+G))-delta)/phi
            data_objective[:, 0, :, 0] = score
            check = np.isnan(data_objective)
            if np.any(check):
                data_objective[check] = 0 # Default Value = 0
                mess = '~¡!~'*8+"\n\tSeveral time series have 'nans'!\n\tWe have replaced those nan values with 'zeros'!\n"+'~¡!~'*8
                print(mess)
            self.data_objective = data_objective
            if rule_index == 0:
                data_rule_objective = np.full((len(self.rules), self.data_objective.shape[0], self.data_objective.shape[1], self.data_objective.shape[2], self.data_objective.shape[3]), np.nan)
            data_rule_objective[rule_index, ...] = data_objective
            if rule_index != len(self.rules)-1:
                self.time_mini = self._time_mini # Welfare purpose!
                self.time_maxi = self._time_maxi # Welfare purpose!
                self.time_delta = self._time_delta # Welfare purpose!
            self.data = np.array(None) # Welfare purpose!
            # Data Processor! [Final]
            # Data View! [Start]
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
                    plt.title(f'Simulation\n{simulation}\nRule\n{rule_index}')
                    plt.xlabel(self.time_unit)
                    plt.ylabel('Score')
                    plt.xlim(self._time_mini, self._time_maxi)
                    plt.ylim(-0.1, 1.1)
                    plt.grid(linestyle = '--')
                    plt.show()
            # Data View! [Final]
        # Data Postprocessor! [Start]
        if self.verbose: print(f"{'~'*8} All Rules! {list(range(len(self.rules)))} : {len(self.rules)} {'~'*8}")
        self.metamorphose_data() # Welfare purpose!
        score_alp = data_rule_objective[0]
        score_bet = data_rule_objective[1]
        score_chi = np.maximum((score_alp+score_bet)/2-np.power(score_alp-score_bet, 2), 0)
        self.data_objective = score_chi.reshape(data_objective.shape)
        if self.verbose or self.show:
            for simulation in range(simulations):
                if simulation >= self.simulations_maxi:
                    break
                plt.plot(x, self.data_objective[simulation, 0, :, 0], color = 'tab:purple')
                plt.title(f'Simulation\n{simulation}\nScore\nAll Rules!')
                plt.xlabel(self.time_unit)
                plt.ylabel('Score')
                plt.xlim(self._time_mini, self._time_maxi)
                plt.ylim(-0.1, 1.1)
                plt.grid(linestyle = '--')
                plt.show()
        # Data Postprocessor! [Final]
        return self
    
    def appraise(self, **keywords):
        check = self.data_objective is None
        mess = f'Please, we must synthesize (or execute/apply) the inference procedure!\ndata_objective = {self.data_objective}'
        assert not check, mess
        perks = keywords.get('perks', [10, 50, 90])
        percentiles = np.percentile(a = self.data_objective, q = perks, axis = 0).reshape((len(perks), -1)).T
        alp = percentiles[:, 1]
        bet = np.power(percentiles[:, 2] - percentiles[:, 0], 2)
        chi = np.maximum(alp - bet, 0)
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

