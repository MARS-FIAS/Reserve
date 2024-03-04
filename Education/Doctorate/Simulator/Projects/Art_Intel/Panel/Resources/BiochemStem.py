######## Name
######## BiochemStem

######## Requires
######## {Modules}

import re
import numpy as np
import numba

class BiochemStem:
    
    """Class 'BiochemStem' Illustration!
    It is an essential part of the 'BiochemSimul' class!
    
    ########
    Attributes
        initial_state: 'dict'
            A dictionary containing the initial state of the given biochemical system.
            It presents every biochemical species involved in the system, together with its initial copy number.
            Convention: upper-case letter for each reactant/product, non-negative integer for each initial copy number.
            Example: initial_state = {'A': 10, 'Z': 0}.
        rates: 'dict'
            A dictionary containing every reaction rate for the given biochemical system.
            Please, use the following convention: lower-case letter and non-negative integer index for each rate constant, e.g. 'k10'.
            Each dictionary (key, value) pair must be like ('str', 'float').
            Example: rates = {'k1': 0.1, 'k2': 2}.
    
    ########
    Methods
        add_reaction(name, prop_fun, delta, equation = None)
            It adds a (new) biochemical reaction to the current system.
            Please, read carefully its illustration!
        del_reaction(name)
            It deletes a reaction from the current biochemical system based on the 'name' argument.
        assemble(show = True)
            It assembles our biochemical system; it makes it ready for simulation/analysis.
    re.sub(s, f'species[{i}]', prop_funs[j])
    """
    
    def __init__(self, initial_state, rates, verbose = False):
        """It initializes our new biochemical system.
        The only necessary initialization attributes are 'initial_state' and 'rates'.
        Also, it instantiates an empty dictionary as a container for all the reactions.
        The (private) instance variable '_assembled' keeps track of the instance object construction status.
        In other words, whenever a reaction is added to or deleted from the current biochemical system, we will need to construct/assemble it (again) before analyzing it.
        
        """
        self.initial_state = initial_state
        self.rates = rates
        self.reactions = {} # Empty dictionary (obvious)
        self._assembled = False # Flag # Instance variable (private)
        self.verbose = verbose
    
    def __repr__(self):
        portrait = f'<{self.__class__.__name__}({self.initial_state}, {self.rates})>'
        return portrait
    
    def __str__(self):
        portrait = repr(self)
        return portrait
    
    def _consistency(self, temp, checkers, verbose):
        for key, value in self.initial_state.items():
            express = '{0} = {1}'.format(key, value)
            exec(express)
        for key, value in self.rates.items():
            express = '{0} = {1}'.format(key, value)
            exec(express)
        _consistent = {key: False for key in checkers} # User info # Default
        _err = 'Unexpected error!\nTry again...' # Default error message
        _express = "print('Is <{0}> consistent?\t{1}'.format(checker, _consistent[checker]))"
        for checker in checkers:
            check = temp[checker]
            if checker == 'prop_fun':
                try:
                    _ = eval(check)
                except NameError as err:
                    _message = "'{}' is using some species/rates not yet defined".format(checker)
                    _err = 'Error!\n{0}\n{1}'.format(err, _message)
                    print(_err)
                except SyntaxError as err:
                    _message = "'{}' is not a valid Python expression".format(checker)
                    _err = 'Error!\n{0}\n{1}'.format(err, _message)
                    print(_err)
                except:
                    print(_err)
                else:
                    _consistent[checker] = True
                finally:
                    exec(_express) if verbose else None
            if checker == 'delta':
                try:
                    _ = {self.initial_state[key] for key in check}
                except KeyError as err:
                    _message = "'{0}' is trying to change some unknown species: {1}".format(checker, err)
                    _err = 'Error!\n{}'.format(_message)
                    print(_err)
                except:
                    print(_err)
                else:
                    _consistent[checker] = True
                finally:
                    exec(_express) if verbose else None
        return _consistent
    
    def add_reaction(self, name, prop_fun, delta, equation = None, verbose = False, jump_diffuse = None):
        """It adds a (new) biochemical reaction to the current system.
        Please, read carefully its illustration!
        
        ########
        Arguments
            name: 'str'
                It provides a name for the new reaction.
                Please, use the following convention: lower-case letter and non-negative integer index for the name, e.g. 'r10'.
            prop_fun: 'str'
                It provides the necessary propensity function for the new reaction.
                It must be of type 'str', and it must represent a valid Python expression.
                Please, be careful and consistent with your definitions!
                Example: prop_fun = 'A*k1'.
            delta: 'dict'
                It presents the discrete increase/reduction in copy number for each affected (product) species.
                Each dictionary (key, value) pair must be like ('str', 'int').
                Example: delta = {'A': -10, 'Z': 2}.
            equation: 'str', 'optional'
                    default = None
                It provides an equation (optional) for the new reaction.
                Example: equation = 'Reactants-k10->Products'.
                Note: the argument 'equation' is not compulsory, but it will be used in a future definiton of the 'BiochemStem' class.
        
        ########
        Returns
            self.reactions: 'dict'
            It returns a dictionary with an entry (biochemical reaction) added to the current system.
        
        """
        _temp = '{' + "'{0}': {0}, '{1}': {1}, '{2}': {2}".format('prop_fun', 'delta', 'jump_diffuse') + '}'
        temp = eval(_temp)
        _consistent = self._consistency(temp, checkers = ['prop_fun', 'delta'], verbose = verbose)
        if not all(_consistent.values()):
            raise RuntimeError('The new reaction is wrong! Check it again!') # Stop!
        self.reactions[name] = temp
        if verbose:
            message = "We have successfully added '{}'!".format(name)
            print(message)
        self._assembled = False # Enforce flag
        return self
    
    def del_reaction(self, name):
        """It deletes a reaction from the current biochemical system based on the 'name' argument.
        
        ########
        Arguments
            name: 'str'
                We remove the reaction given by 'name' from the current biochemical system.
                It must be consistent with the argument 'name' provided when the method 'add_reaction' was invoked.
        
        ########
        Returns
            self.reactions: 'dict'
            It returns a dictionary with an entry (biochemical reaction) deleted from the current system.
        
        """
        self.reactions.pop(name)
        message = "We have successfully deleted '{}'!".format(name)
        print(message)
        self._assembled = False # Enforce flag
        return self
    
    def _delta_mat(self, rows, cols):
        delta_mat = np.full(shape = (len(rows), len(cols)), fill_value = np.nan, dtype = np.int32)
        it = np.nditer(op = delta_mat, flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = len(cols))
        for d in it:
            _i = divmod(it.iterindex, len(cols))
            i = _i[0]
            if i == 0:
                d[...] = [0] * len(cols) # d[...] = list(self.initial_state.values())
            else:
                _j = self.reactions[rows[i]]['delta']
                j = [(_j[key] if key in _j else 0) for key in cols.values()]
                d[...] = j
        return delta_mat
    
    def _jump_diffuse_delta_mat(self, rows, cols, bove):
        jump_diffuse_delta_mat = np.full(shape = (len(rows), len(cols)), fill_value = 0, dtype = np.int32)
        for j in np.argwhere(bove).flatten():
            _d = self.reactions[rows[j+1]]['jump_diffuse']
            d = [(_d[key] if key in _d else 0) for key in cols.values()]
            jump_diffuse_delta = np.array(d, np.int32)
            jump_diffuse_delta_mat[j+1, :] = jump_diffuse_delta
        return jump_diffuse_delta_mat
    
    def _translator(self):
        rates = list(self.rates.items())
        species = list(self.assembly['species'].items())
        _prop_funs = self.assembly['prop_funs'].copy()
        for j in range(len(_prop_funs)):
            if self.verbose:
                print(_prop_funs[j])
            for i in range(len(species)):
                s = '\\b' + species[i][1] + '\\b'
                flag = re.search(s, _prop_funs[j])
                if flag is not None:
                    new = re.sub(s, f'species[{i}]', _prop_funs[j])
                    _prop_funs[j] = new
                    if self.verbose:
                        print(new)
            for i in range(len(rates)):
                r = '\\b' + rates[i][0] + '\\b'
                flag = re.search(r, _prop_funs[j])
                if flag is not None:
                    new = re.sub(r, f'rates[{i}]', _prop_funs[j])
                    _prop_funs[j] = new
                    if self.verbose:
                        print(new)
            if self.verbose:
                print('')
            _pile = _prop_funs.copy()
            for index in range(len(_pile)):
                _pile[index] = f'\n    if index == {index}:\n        return {_prop_funs[index]}'
            pile = ''.join(_pile)
            label = 'prop_funs' # Function Name!
            template = f'@numba.njit\ndef {label}(index, rates, species):{pile}\n    else:\n        return None'
            exec(template)
            exec(f'self.{label} = {label}')
            exec(f'self._{label} = template')
        if self.verbose:
            print(template)
        return self
    
    def assemble(self, show = False):
        """It assembles our biochemical system; it makes it ready for simulation/analysis.
        
        ########
        Arguments
            show: 'bool'
                    default = True
                Do we want to see the assembled system?
                All the (biochemical) components will be shown in a cohesive way.
        
        ########
        Returns
            self.assembly: 'dict'
                Keys
                    'species': dictionary enumerating each species name.
                    'reactions': dictionary enumerating each reaction name.
                    'prop_funs': list containing propensity functions for all reactions.
                    'delta_mat': a matrix representing the delta for every reaction.
            It returns essential components for simulation/analysis.
        
        """
        species = dict(enumerate(self.initial_state)) # species = dict(zip(self.initial_state, range(len(self.initial_state))))
        reactions = dict(enumerate(['r0', *self.reactions])) # reactions = dict(zip(['r0', *self.reactions], range(len(self.reactions)+1)))
        prop_funs = ['0']
        prop_funs.extend([self.reactions[key]['prop_fun'] for key in self.reactions])
        delta_mat = self._delta_mat(rows = reactions, cols = species)
        jump_diffuse_vet = np.array([self.reactions[key]['jump_diffuse'] for key in self.reactions], np.bool_)
        if jump_diffuse_vet.any():
            jump_diffuse_delta_mat = self._jump_diffuse_delta_mat(rows = reactions, cols = species, bove = jump_diffuse_vet)
        else:
            jump_diffuse_delta_mat = None
        self.assembly = {'species': species, 'reactions': reactions, 'prop_funs': prop_funs, 'delta_mat': delta_mat, 'jump_diffuse_vet': jump_diffuse_vet, 'jump_diffuse_delta_mat': jump_diffuse_delta_mat}
        self._assembled = True # Enforce flag
        #
        self._translator()
        #
        if show:
            print(str(self))
        else:
            pass # print("\n") # print(repr(self))
        return self
