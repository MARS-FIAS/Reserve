###########################
######## Utilities ########
###########################

#%%# Catalyzer

import numpy as np
import torch
from torch.distributions import Independent, Uniform
import matplotlib
import matplotlib.pyplot as plt

#%%# Make Paras

def make_paras(para_set_raw = None, para_set_mode = None, verbose = False):
    descript = "{'Parameter': ('Act_Value', ('Mini_Value', 'Maxi_Value'), 'Uncountable_XOR_Countable', 'Time_Unit'), 'Mode': {0, 1}}"
    if verbose: print(descript)
    if para_set_raw is not None:
        mess = "The variable 'para_set_raw' must be a valid dictionary!"
        check = type(para_set_raw) is dict
        assert check, mess
        mess = 'Every dictionary key must be a string!'
        _check = [type(key) is str for key in para_set_raw.keys()]
        check = all(_check)
        assert check, mess
        mess = 'Every dictionary value must be a quartet!'
        _check = [type(value) is tuple and len(value) == 4 for value in para_set_raw.values()]
        check = all(_check)
        assert check, mess
        mess = "Every quartet must specify (position 0) a valid 'true' parameter value and (position 1) a valid 'duple' parameter range!"
        _check = [value[1][0] <= value[0] <= value[1][1] for value in para_set_raw.values()]
        check = all(_check)
        assert check, mess
        mess = "Every quartet must specify (position 2) a valid 'Uncountable XOR Countable' flag! {0: 'Uncountable', 1: 'Countable'}"
        _check = [value[2] in [0, 1] for value in para_set_raw.values()]
        check = all(_check)
        assert check, mess
        _mess = "Every quartet must specify (position 2) a valid 'Uncountable XOR Countable' flag compatible with a valid 'para_set_mode' argument!"
        mess = _mess + "\n('Uncountable_XOR_Countable', 'para_set_mode')\n\t'Compatible' = {(0, 0), (0, 1), (1, 0)}\n\t'Incompatible' = {(1, 1)}"
        _check = [not(value[2] and bool(para_set_mode)) for value in para_set_raw.values()]
        check = all(_check)
        assert check, mess
        mess = "Every quartet must specify (position 3) a valid 'Time Unit' flag! {0: 'Seconds', 1: 'Minutes', 2: 'Hours', None: 'None'}"
        _check = [value[3] in [0, 1, 2, None] for value in para_set_raw.values()]
        check = all(_check)
        assert check, mess
        para_set_true = para_set_raw
        if para_set_mode == 0:
            para_set = np.array([value[0] for value in para_set_true.values()])
        elif para_set_mode == 1:
            para_set = np.array([(value[0]-value[1][0])/(value[1][1]-value[1][0]) for value in para_set_true.values()])
        else:
            mess = f"Invalid mode!\n\t'Mode = {para_set_mode}'"
            raise RuntimeError(mess)
    else:
        para_set_true = None
        para_set = None
    paras = (para_set, para_set_true)
    return paras

#%%# Prior Mixer Class

class PriorMixer(Independent):
    
    def __init__(self, low = 0, high = 1, reinterpreted_batch_ndims = 1, **keywords):
        super().__init__(
            Uniform(
                low = torch.as_tensor(data = low, dtype = torch.float32),
                high = torch.as_tensor(data = high, dtype = torch.float32),
                validate_args = False
            ),
            reinterpreted_batch_ndims
        )
        self.para_set_true = keywords.get('para_set_true', None)
        if self.para_set_true is None:
            self.para_set_flags = torch.tensor(data = [0]*self.base_dist.batch_shape[0], dtype = torch.bool)
        else:
            self.para_set_flags = torch.tensor(data = [value[2] for value in self.para_set_true.values()], dtype = torch.bool)
    
    def sample(self, sample_shape = torch.Size()):
        ret = self.base_dist.sample(sample_shape)
        if torch.any(self.para_set_flags):
            if ret.ndim == 0:
                pass
            elif ret.ndim == 1:
                ret[self.para_set_flags] = torch.round(input = ret[self.para_set_flags])
            else: # ret.ndim > 1
                ret[:, self.para_set_flags] = torch.round(input = ret[:, self.para_set_flags])
        return ret

#%%# Make Prior Mixer

def make_prior_mixer(para_set_true, para_set_mode, para_set_sieve = None, verbose = False):
    _para_set_true = para_set_true.copy() # Welfare purpose!
    para_set_keys = list(_para_set_true.keys())
    if para_set_sieve is None:
        if verbose: print(f"All 'para_set'!\n\t{para_set_keys}")
    else:
        check = [para in para_set_keys for para in para_set_sieve]
        mess = f"The list 'para_set_sieve' is not consistent with the 'para_set_true' dictionary!\n\t{np.array(para_set_sieve)[np.invert(check)]}"
        assert all(check), mess
        for para in para_set_keys:
            if para in para_set_sieve:
                if verbose: print(f'Save!\t{para}')
            else:
                _ = _para_set_true.pop(para)
                if verbose: print(f'Kill!\t\t{para}\t\t{_}')
        if verbose: print(f"Sieve 'para_set'!\n\t{para_set_sieve}")
    if para_set_mode == 0:
        rho = 0.5 # Constant!
        epsilon = 1e-5 # Adjustable! # epsilon > 0 # epsilon << 1
        para_span_low = torch.tensor([value[1][0]-rho+epsilon if value[2] else value[1][0] for value in _para_set_true.values()]) # Parameter Range (Low)
        para_span_high = torch.tensor([value[1][1]+rho-epsilon if value[2] else value[1][1] for value in _para_set_true.values()]) # Parameter Range (High)
        prior_mixer = PriorMixer(low = para_span_low, high = para_span_high, para_set_true = para_set_true)
    elif para_set_mode == 1:
        para_span = torch.tensor([0, 1]) # Parameter Range
        card = len(_para_set_true) # Parameter Set Cardinality
        prior_mixer = PriorMixer(low = para_span[0]*torch.ones(card), high = para_span[1]*torch.ones(card), para_set_true = para_set_true)
    else:
        mess = f"Invalid mode!\n\t'Mode = {para_set_mode}'"
        raise NotImplementedError(mess)
    para_set_true = _para_set_true.copy() # Welfare purpose!
    if verbose: print(f'Prior Mixer!\n\t{prior_mixer}')
    return prior_mixer

#%%# Instate Para Set

def instate_para_set(proposal):
    initial_para_set = proposal.sample()
    return initial_para_set

#%%# Move Para Set

def move_para_set(current_para_set, iota, prior, prior_mixer = None, verbose = False, **keywords):
    if type(iota) is float:
        if not(0 < iota < 1):
            mess = "Invalid 'iota' value!\n\t'iota: (0, 1)'"
            raise RuntimeError(mess)
    elif type(iota) is torch.Tensor:
        if iota.size() != current_para_set.size():
            mess = "Invalid 'iota' tensor!\n\tThe size of 'iota' and the size of 'current_para_set' must be equal!"
            raise RuntimeError(mess)
        elif not torch.all(torch.logical_and(0 < iota, iota < 1)):
            mess = "Invalid 'iota' values!\n\tAll! 'iota: (0, 1)'"
            raise RuntimeError(mess)
    else:
        mess = 'Oops! Something went wrong!'
        raise RuntimeError(mess)
    para_span_low = prior.support.base_constraint.lower_bound # Parameter Range (Low)
    para_span_high = prior.support.base_constraint.upper_bound # Parameter Range (High)
    para_span = para_span_high - para_span_low
    para_span_delta = iota * para_span
    para_span_delta_low = torch.maximum(para_span_low, current_para_set - para_span_delta)
    para_span_delta_high = torch.minimum(para_span_high, current_para_set + para_span_delta)
    if prior_mixer is not None:
        rho = 0.5 # Constant!
        epsilon = 1e-5 # Adjustable! # epsilon > 0 # epsilon << 1
        para_span_delta_low[prior_mixer.para_set_flags] = para_span_delta_low[prior_mixer.para_set_flags] - rho + epsilon
        para_span_delta_high[prior_mixer.para_set_flags] = para_span_delta_high[prior_mixer.para_set_flags] + rho - epsilon
        para_set_true = prior_mixer.para_set_true
    else:
        para_set_true = None
    mover = PriorMixer(low = para_span_delta_low, high = para_span_delta_high, para_set_true = para_set_true)
    if verbose: print(f'Para Span Delta!\n\tLow!\n{para_span_delta_low}\n\tHigh!\n{para_span_delta_high}')
    next_para_set = mover.sample()
    dimes = keywords.get('dimes', None)
    if dimes is not None: # (dimes = None) == (dimes = mover.event_shape[0])
        mess = f"Invalid 'dimes' value!\n\tInteger!\n\t'dimes: (1, ..., {mover.event_shape[0]})'"
        check = (type(dimes) is int) and (1 <= dimes <= mover.event_shape[0])
        assert check, mess
        perm = torch.randperm(mover.event_shape[0])
        keep = perm[0:dimes]
        lose = perm[dimes:len(perm)]
        if verbose:
            _keep = next_para_set.clone()
            _lose = next_para_set.clone()
            _keep[lose] = torch.nan
            _lose[keep] = torch.nan
            print(f"Next Para Set! {next_para_set}\n\t'Keep {dimes}! {_keep}'\n\t'Lose {mover.event_shape[0]-dimes}! {_lose}'")
        next_para_set[lose] = current_para_set[lose]
    if verbose: print(f'Para Set!\n\tCurrent!\n{current_para_set}\n\tNext!\n{next_para_set}')
    return next_para_set

#%%# Simulation Procedure: Synopsis Preparation!

def make_synopsis(explorations, para_set_true, verbose = False):
    synopsis_varas = list(para_set_true.keys())+['Register'] # {'Register': {'Current_Meta_Score', 'Optimum_Meta_Score'}}
    synopsis_size = (2, explorations, len(synopsis_varas)) # {0: 'Current_Meta_Score', 1: 'Optimum_Meta_Score'}
    synopsis = torch.full(synopsis_size, torch.nan)
    if verbose: print(f'Synopsis!\t{synopsis.size()}')
    return synopsis

def refresh_synopsis(synopsis, exploration, roster_varas = tuple(), verbose = False):
    current_para_set, current_meta_score, optimum_para_set, optimum_meta_score = roster_varas
    synopsis[0, exploration, :] = torch.cat((current_para_set, current_meta_score.reshape(tuple([1]))))
    synopsis[1, exploration, :] = torch.cat((optimum_para_set, optimum_meta_score.reshape(tuple([1]))))
    return synopsis

def save_synopsis(synopsis, task_pin, task, path, tag):
    label = path+tag+'_Synopsis_'+str(task_pin)+'_'+str(task)+'.pt'
    torch.save(synopsis, label)
    print(f'Save Task Data! {tag}\t{task_pin} ~ {task}')
    return None

#%%# Synopsis Data Load!

def synopsis_data_load(task_pins_info, path, tag):
    import os
    import re
    if task_pins_info is None:
        _arcs = os.listdir(path)
        arcs = [arc for arc in _arcs if re.findall(tag+'_', arc)]
        arcs_discovery = [arc for arc in arcs if re.findall('_Synopsis_', arc)]
        _arcs_discovery_identify = [re.findall('(\d+)_(\d+)\.pt', arc)[0] for arc in arcs_discovery]
        arcs_discovery_identify = [(int(ident[0]), int(ident[1])) for ident in _arcs_discovery_identify]
        arcs_discovery_identify.sort()
        task_pins = list(set([ident[0] for ident in arcs_discovery_identify]))
        task_pins.sort()
        tasks_info = {task_pin: [ident[1] for ident in arcs_discovery_identify if task_pin == ident[0]] for task_pin in task_pins}
        mess = 'Oops! Something went wrong!'
        check = task_pins == list(tasks_info.keys())
        assert check, mess
        task_pins_mini = min(task_pins)
        task_pins_maxi = max(task_pins)
    elif type(task_pins) is tuple:
        task_pins_mini = task_pins_info[0]
        task_pins_maxi = task_pins_info[1]
        task_pins = range(task_pins_mini, task_pins_maxi+1)
    else:
        mess = "The format is invalid! The variable 'task_pins_info' must be either 'None' or 'tuple'!"
        raise RuntimeError(mess)
    print(f"Load Synopsis Data! '{tag}' Task Pins! {task_pins_mini} : {task_pins_maxi} Total {len(task_pins)}")
    for task_pin in task_pins:
        tasks = tasks_info[task_pin]
        if task_pin == task_pins_mini:
            synopses = list()
        for task in tasks:
            label = path+tag+'_Synopsis_'+str(task_pin)+'_'+str(task)+'.pt'
            _synopses = torch.load(label)
            synopses.append(_synopses)
            print(f'Task Pin! {task_pin} Tasks! {task+1} ~ {len(tasks)} Index {task_pin} {task}')
    print(f"Load Synopsis Data! '{tag}' Task Pins! {task_pins_mini} : {task_pins_maxi} Total {len(task_pins)}")
    collect = (synopses, tasks_info)
    return collect

#%%# Section [New]


