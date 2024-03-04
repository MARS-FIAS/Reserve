###########################
######## Utilities ########
###########################

#%%# Catalyzer

# import re
import numpy as np
# import numba
from scipy import interpolate
import torch
from sbi.inference import simulate_for_sbi
import time

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Interpolate [State Tor Subset]

def interpolator(simul, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, kind = 0, sup_comp = False, verbose = False, **keywords):
    if sup_comp: verbose = False # Enforce!
    noms = {'Seconds': 0, 'Minutes': 1, 'Hours': 2}
    epoch_mini = [(_nom, int(np.nanmin(simul.epoch_mat)/np.power(60, nom))) for _nom, nom in noms.items()]
    if verbose: print('Mini Epoch\n\t', epoch_mini)
    epoch_maxi = [(_nom, int(np.nanmax(simul.epoch_mat)/np.power(60, nom))) for _nom, nom in noms.items()]
    if verbose: print('Maxi Epoch\n\t', epoch_maxi)
    _nom = noms[time_unit]
    nom = np.power(60, _nom)
    xl = time_mini
    xr = time_maxi*nom
    ix = np.arange(xl, xr+nom, time_delta*nom)
    ix_maxi = [(_nom, int(xr/np.power(60, nom))) for _nom, nom in noms.items()]
    if verbose: print('Maxi Interpolation Epoch\n\t', ix_maxi)
    values = list(simul.stem.assembly['species'].values()) if not hasattr(simul, 'species_objective') else simul.species_objective
    _s = species
    s = [values.index(value) for value in _s]
    cells = simul.cells
    data_ix = ix
    data_iy = np.full((len(s), len(ix), cells), np.nan)
    for i in range(cells):
        for j in s:
            _x = simul.epoch_mat[:, i]
            where = np.isnan(_x)
            here = np.argmax(where) if np.any(where) else len(_x)
            x = _x[0:here]
            _y = simul.state_tor[:, j, i] if not hasattr(simul, 'state_tor_objective') else simul.state_tor_objective[:, j, i]
            y = _y[0:here]
            fun = interpolate.interp1d(x = x, y = y, kind = kind, bounds_error = keywords.get('err', True), fill_value = keywords.get('fill_value', np.nan))
            iy = fun(ix)
            data_iy[s.index(j), :, i] = iy
    data = (data_ix, data_iy)
    return data

#%%# Decimator [Interpolation]

def decimator():
    return None

#%%# Plot [Interpolation]

def plotful(data, species, time_unit):
    ix, iy = data
    noms = {'Seconds': 0, 'Minutes': 1, 'Hours': 2}
    _nom = noms[time_unit]
    nom = np.power(60, _nom)
    cells = iy.shape[2]
    x = ix/nom
    x_mini = 1*np.nanmin(x)
    x_maxi = 1.025*np.nanmax(x)
    y_mini = 1*np.nanmin(iy)
    y_maxi = 1.025*np.nanmax(iy)
    for cell in range(cells):
        y = iy[:, :, cell].T
        plt.plot(x, y, drawstyle = 'steps-post', linestyle = '-')
        plt.title(f'Cell\n{cell}')
        plt.xlim(x_mini, x_maxi)
        plt.ylim(y_mini, y_maxi)
        plt.xlabel(time_unit)
        plt.ylabel('Copy Number')
        plt.legend(species)
        plt.grid(linestyle = '--')
        plt.show()
    return None

#%%# Previewer [Restructure]

def previewer(data, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, sup_comp = False, verbose = False):
    check = len(data.shape) # (simulations, len(species), len(x), cells)
    mess = "Please, we must restructure/reshape the data!\n\t'shape = (simulations, len(species), len(x), cells)'"
    assert check == 4, mess
    simulations = data.shape[0]
    cells = data.shape[3] # data.shape[-1]
    x = np.arange(time_mini, time_maxi+1, time_delta)
    for simulation in range(simulations):
        if simulation >= simulations_maxi:
            break
        for cell in range(cells):
            y = data[simulation, :, :, cell].T
            plt.plot(x, y, drawstyle = 'steps-post', linestyle = '-')
            plt.title(f'Simulation ~ Cell\n{simulation} ~ {cell}')
            plt.xlabel(time_unit)
            plt.ylabel('Copy Number')
            plt.legend(species)
            plt.grid(linestyle = '--')
            plt.show()
    return None

#%%# Restructure [Simulation Data]

def restructure(data, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, sup_comp = False, verbose = False):
    
    if sup_comp: verbose = False # Enforce!
    
    noms = {'Seconds': 0, 'Minutes': 1, 'Hours': 2}
    _nom = noms[time_unit]
    nom = np.power(60, _nom)
    xl = time_mini*nom
    xr = time_maxi*nom
    x = np.arange(xl, xr+nom, time_delta*nom)/nom
    
    if len(data.shape) == 1:
        simulations = 1
        take = 0
    else: # len(data.shape) == 2
        simulations = data.shape[0]
        take = 1
    
    _cells = data.shape[take]/len(species)
    cells = int(_cells/len(x))
    
    data_rest = data.reshape((simulations, len(species), len(x), cells))
    
    if verbose:
        previewer(data_rest, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose)
    
    return data_rest

#%%# Instate [State Tor]

def instate_state_tor(stem, cell_layers, layer_cells, state_tors, pat_mode = None, initial_pattern = None, blank = False):
    if blank:
        state_tor = None
    else:
        cells = cell_layers * layer_cells + 1
        cells_ICM = cells - 1
        cells_EPI = cells_ICM - layer_cells
        cells_PRE = cells_ICM - cells_EPI
        state_tor_PRE, state_tor_EPI, state_tor_V = state_tors.values()
        for cell in range(cells):
            exec(f'_state_tor_{cell} = stem.initial_state.copy()')
        if initial_pattern is None:
            for cell in range(cells):
                if cell in range(cells_PRE):
                    exec(f'_state_tor_{cell}.update(state_tor_PRE)')
                elif cell in range(cells_PRE, cells_PRE + cells_EPI):
                    exec(f'_state_tor_{cell}.update(state_tor_EPI)')
                else:
                    exec(f'_state_tor_{cell}.update(state_tor_V)')
        else: # initial_pattern is not None
            if pat_mode == 'Harsh':
                for cell in range(cells):
                    if cell in range(cells_ICM):
                        if initial_pattern[cell] == 1: # G
                            exec(f'_state_tor_{cell}.update(state_tor_PRE)')
                        elif initial_pattern[cell] == 0: # N
                            exec(f'_state_tor_{cell}.update(state_tor_EPI)')
                    else:
                        exec(f'_state_tor_{cell}.update(state_tor_V)')
            elif pat_mode == 'Slide':
                state_tor_ICM = {key: initial_pattern*state_tor_PRE[key] if key != 'EI' else state_tor_PRE[key] for key in state_tor_PRE.keys()} # state_tor_EPI
                for cell in range(cells):
                    if cell in range(cells_ICM):
                        exec(f'_state_tor_{cell}.update(state_tor_ICM)')
                    else:
                        exec(f'_state_tor_{cell}.update(state_tor_V)')
            elif pat_mode == 'IPA':
                for cell in range(cells):
                    if cell == initial_pattern['aim']:
                        state_tor_NULL = initial_pattern['null']
                        exec(f'_state_tor_{cell}.update(state_tor_NULL)')
                    elif cell in range(cells_PRE):
                        exec(f'_state_tor_{cell}.update(state_tor_PRE)')
                    elif cell in range(cells_PRE, cells_PRE + cells_EPI):
                        exec(f'_state_tor_{cell}.update(state_tor_EPI)')
                    else:
                        exec(f'_state_tor_{cell}.update(state_tor_V)')
            elif pat_mode == 'SAP':
                for cell in range(cells):
                    if cell < cells_ICM:
                        if initial_pattern[cell] == 1: # G
                            exec(f'_state_tor_{cell}.update(state_tor_PRE)')
                        elif initial_pattern[cell] == 0: # N
                            exec(f'_state_tor_{cell}.update(state_tor_EPI)')
                    else:
                        exec(f'_state_tor_{cell}.update(state_tor_V)')
            elif pat_mode == 'Fish_Bind':
                for cell in range(cells):
                    if cell in range(cells_PRE):
                        state_tor_PRE.update(initial_pattern[cell])
                        exec(f'_state_tor_{cell}.update(state_tor_PRE)')
                    elif cell in range(cells_PRE, cells_PRE + cells_EPI):
                        state_tor_EPI.update(initial_pattern[cell])
                        exec(f'_state_tor_{cell}.update(state_tor_EPI)')
                    else:
                        exec(f'_state_tor_{cell}.update(state_tor_V)')
            else:
                mess = f"Oops! The pattern mode '{pat_mode}' is unknown!"
                raise RuntimeError(mess)
        state_tors = ''
        for cell in range(cells):
            exec(f'state_tor_{cell} = np.array(list(_state_tor_{cell}.values())).reshape(-1, 1)')
            if cell == 0:
                state_tors += f'(state_tor_{cell}, '
            if 0 < cell < cells - 1:
                state_tors += f'state_tor_{cell}, '
            if cell == cells - 1:
                state_tors += f'state_tor_{cell})'
        state_tor = eval(f'np.concatenate({state_tors}, 1)')
    return state_tor

#%%# Instate [Rate Mat]

def instate_rate_mat(stem, cells, parameter_set, parameter_set_true, para_fun, rates_exclude, rho_mat, sigma_vet, blank = False, **keywords):
    if blank:
        rate_mat = None
    else:
        kd_FM = keywords.get('kd_FM', stem.rates['kd_FM'])
        # FM * kd_FM
        kd_FL = keywords.get('kd_FL', stem.rates['kd_FL']) # td_FL
        kd_FS_UP = keywords.get('kd_FS_UP', stem.rates['kd_FS_UP']) # td_FS(|_UP)
        kd_FS_DO = keywords.get('kd_FS_DO', stem.rates['kd_FS_DO']) # td_FS(|_DO)
        # FL * kd_FL # FS_UP * kd_FS_UP # FS_DO * kd_FS_DO
        ksig_C = keywords.get('ksig_C', stem.rates['ke_F_CM']) # tau_C
        ksig_M = keywords.get('ksig_M', stem.rates['kjd_F_MM']) # tau_M
        ksig_L = keywords.get('ksig_L', stem.rates['kjd_F_LM']) # tau_L
        ksig_S_UP = keywords.get('ksig_S_UP', stem.rates['ke_FS_UP_DO']) # tau_S(|_UP)
        ksig_S_DO = keywords.get('ksig_S_DO', stem.rates['ke_FS_DO_UP']) # tau_S(|_DO)
        # ksig_C = ke_F_CM + kjd_F_CM + kjd_F_CL + kjd_F_CS_UP + kjd_F_CS_DO # ksig_M = kjd_F_MM + kjd_F_ML + kjd_F_MS_UP + kjd_F_MS_DO # ksig_L = kjd_F_LM + ke_F_LS_UP + ke_F_LS_DO # ksig_S_UP = kjd_F_SM_UP + ke_F_SL_UP + ke_FS_UP_DO # ksig_S_DO = kjd_F_SM_DO + ke_F_SL_DO + ke_FS_DO_UP
        chi_auto = 0.5
        chi_para = 1 - chi_auto
        # chi_auto + chi_para = 1
        lad_mem = 1/3 # 'Sphere' Abstraction
        lad_sink = 2/3 # 'Sphere' Abstraction
        # lad_mem + lad_sink = 1
        sigma_mem = sigma_vet[0] # 'Sphere AND Disk' Abstraction
        sigma_lumen = sigma_vet[0] # 'Sphere AND Disk' Abstraction
        sigma_sink = sigma_vet[1] # 'Sphere AND Disk' Abstraction
        # sigma_(mem|lumen) + sigma_sink = 1
        rate_mat_V = {
            'kjd_F_LM': lad_mem*ksig_L, 'ke_F_LS_UP': 0, 'ke_F_LS_DO': lad_sink*ksig_L,
            'kjd_F_SM_UP': sigma_mem*ksig_S_UP, 'ke_F_SL_UP': 0, 'ke_FS_UP_DO': sigma_sink*ksig_S_UP,
            'kjd_F_SM_DO': 0, 'ke_F_SL_DO': sigma_lumen*ksig_S_DO, 'ke_FS_DO_UP': sigma_sink*ksig_S_DO
        }
        rate_mat_V.update({'kd_FL': kd_FL, 'kd_FS_UP': kd_FS_UP, 'kd_FS_DO': kd_FS_DO})
        rate_mat_V.update({key: 0 for key in stem.rates.keys() if key not in rates_exclude})
        for cell in range(cells):
            exec(f'_rate_mat_{cell} = stem.rates.copy()')
            if cell != cells - 1:
                rho_mem, rho_lumen, rho_sink = rho_mat[cell]
                rate_mat_ICM = {
                    'ke_F_CM': chi_auto*ksig_C,
                    'kjd_F_CM': chi_para*rho_mem*ksig_C, 'kjd_F_CL': chi_para*rho_lumen*ksig_C, 'kjd_F_CS_UP': chi_para*rho_sink*ksig_C, 'kjd_F_CS_DO': 0,
                    'kjd_F_MM': rho_mem*ksig_M, 'kjd_F_ML': rho_lumen*ksig_M, 'kjd_F_MS_UP': rho_sink*ksig_M, 'kjd_F_MS_DO': 0,
                    'kjd_F_LM': 0, 'kjd_F_SM_UP': 0, 'kjd_F_SM_DO': 0
                }
                rate_mat_ICM.update({'kd_FM': kd_FM})
                rate_mat_ICM.update({key: 0 for key in stem.rates.keys() if key in rates_exclude})
                exec(f'_rate_mat_{cell}.update(rate_mat_ICM)')
            else:
                exec(f'_rate_mat_{cell}.update(rate_mat_V)')
        rate_mats = ''
        for cell in range(cells):
            exec(f'rate_mat_{cell} = np.array(list(_rate_mat_{cell}.values())).reshape(-1, 1)')
            if cell == 0:
                rate_mats += f'(rate_mat_{cell}, '
            if 0 < cell < cells - 1:
                rate_mats += f'rate_mat_{cell}, '
            if cell == cells - 1:
                rate_mats += f'rate_mat_{cell})'
        rate_mat = eval(f'np.concatenate({rate_mats}, 1)')
    return rate_mat

#%%# Make Jump Diffuse Tor Simul

def make_jump_diffuse_tor_simul(simul, comm_classes_portrait, jump_diffuse_seed, blank = False):
    if blank:
        jump_diffuse_tor = None
    else:
        check = 'jump_diffuse_tor_old' in globals()
        if not check:
            global jump_diffuse_comm_classes_old
            global comm_class_tor_old
            global comm_class_summary_old
            global jump_diffuse_tor_old
            jump_diffuse_tor = simul.jump_diffuse_assemble(comm_classes_portrait, jump_diffuse_seed)
            jump_diffuse_comm_classes_old = simul.jump_diffuse_comm_classes.copy()
            comm_class_tor_old = simul.comm_class_tor.copy()
            comm_class_summary_old = simul.comm_class_summary.copy()
            jump_diffuse_tor_old = jump_diffuse_tor.copy()
        else:
            simul.jump_diffuse_comm_classes = jump_diffuse_comm_classes_old
            simul.comm_class_tor = comm_class_tor_old
            simul.comm_class_summary = comm_class_summary_old
            jump_diffuse_tor = jump_diffuse_tor_old
    jump_diffuse_tor_simul = (jump_diffuse_tor, simul)
    return jump_diffuse_tor_simul

#%%# Make Paras

def make_paras(para_set_raw = None, para_set_mode = None, verbose = False):
    descript = "{'Parameter': ('Act_Value', ('Mini_Value', 'Maxi_Value')), 'Mode': {0, 1}}"
    if verbose: print(descript)
    if para_set_raw is not None:
        mess = "The variable 'para_set_raw' must be a valid dictionary!"
        check = type(para_set_raw) is dict
        assert check, mess
        mess = 'Every dictionary key must be a string!'
        _check = [type(key) is str for key in para_set_raw.keys()]
        check = all(_check)
        assert check, mess
        mess = 'Every dictionary value must be a duple!'
        _check = [type(value) is tuple and len(value) == 2 for value in para_set_raw.values()]
        check = all(_check)
        assert check, mess
        mess = "Every duple must specify a valid 'true' parameter value and a valid 'duple' parameter range!"
        _check = [value[1][0] <= value[0] <= value[1][1] for value in para_set_raw.values()]
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

#%%# Retrieve Paras

def retrieve_paras(para_set, para_set_true, para_set_mode, verbose = False):
    para_set_act = dict()
    para_keys = list(para_set_true.keys())
    if para_set_mode == 0:
        for para_key in para_keys:
            para_value = para_set[para_keys.index(para_key)]
            para_set_act.update({para_key: para_value})
    elif para_set_mode == 1:
        for para_key in para_keys:
            alp = para_set_true[para_key][1][0]
            bet = para_set_true[para_key][1][1]
            chi = bet - alp
            para_value = chi * para_set[para_keys.index(para_key)] + alp
            para_set_act.update({para_key: para_value})
    else:
        mess = f"Invalid mode!\n\t'Mode = {para_set_mode}'"
        raise RuntimeError(mess)
    if verbose: print(para_set_true)
    return para_set_act

#%%# Make Para Fun [Closure]

def make_para_fun(parameter_set, parameter_set_true, parameter_set_mode):
    _parameter_set_true = list(parameter_set_true.keys())
    parameter_set_index = {key: _parameter_set_true.index(key) for key in _parameter_set_true}
    if parameter_set_mode == 0:
        def para_fun(parameter_key):
            parameter_value = parameter_set[parameter_set_index[parameter_key]]
            return parameter_value
    elif parameter_set_mode == 1:
        def para_fun(parameter_key):
            alp = parameter_set_true[parameter_key][1][0]
            bet = parameter_set_true[parameter_key][1][1]
            chi = bet - alp
            parameter_value = chi * parameter_set[parameter_set_index[parameter_key]] + alp
            return parameter_value
    else:
        mess = f"Invalid mode!\n\t'Mode = {parameter_set_mode}'"
        raise RuntimeError(mess)
    return para_fun

#%%# Make Prior

def make_prior(para_set_true, para_set_mode, para_set_sieve = None, verbose = False):
    from sbi import utils
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
        para_span_low = torch.tensor([value[1][0] for value in _para_set_true.values()]) # Parameter Range (Low)
        para_span_high = torch.tensor([value[1][1] for value in _para_set_true.values()]) # Parameter Range (High)
        prior = utils.BoxUniform(low = para_span_low, high = para_span_high)
    elif para_set_mode == 1:
        para_span = torch.tensor([0, 1]) # Parameter Range
        card = len(_para_set_true) # Parameter Set Cardinality
        prior = utils.BoxUniform(low = para_span[0]*torch.ones(card), high = para_span[1]*torch.ones(card))
    else:
        mess = f"Invalid mode!\n\t'Mode = {para_set_mode}'"
        raise NotImplementedError(mess)
    para_set_true = _para_set_true.copy() # Welfare purpose!
    if verbose: print(f'Prior!\n\t{prior}')
    return prior

#%%# Make Simulator Ready [Partial Object]

def make_simulator_ready(simulator, argument_values):
    import inspect
    from functools import partial
    argument_keys = tuple(inspect.signature(simulator).parameters.keys())[1:len(argument_values)]
    arguments = {argument_keys[index]: argument_values[index] for index in range(len(argument_keys))}
    simulator_ready = partial(simulator, **arguments)
    return simulator_ready

#%%# Simulation Procedure [Local Computer]

def simul_procedure_loa_comp(simulator_ready, prior, tasks, task_simulations, path, tag, safe = False):
    _a = time.time()
    for task in range(tasks):
        theta_set, trajectory_set = simulate_for_sbi(simulator_ready, prior, task_simulations)
        label = path+tag+'_Theta_Set_'+str(task)+'.pt'
        if safe: torch.save(theta_set, label)
        label = path+tag+'_Trajectory_Set_'+str(task)+'.pt'
        if safe: torch.save(trajectory_set, label)
        print(f'Save Task Data! {tag}\t{task+1} ~ {tasks}\tIndex {task}')
    _b = time.time()
    print(f'Simul Time!\t{_b-_a}')
    collect_last = (theta_set, trajectory_set) # Only Last Task!
    return collect_last

#%%# Simulation Procedure [Super Computer]

def simul_procedure_sup_comp(simulator_ready, prior, task, task_simulations, path, tag, safe = False):
    _a = time.time()
    theta_set, trajectory_set = simulate_for_sbi(simulator_ready, prior, task_simulations)
    label = path+tag+'_Theta_Set_'+str(task)+'.pt'
    if safe: torch.save(theta_set, label)
    label = path+tag+'_Trajectory_Set_'+str(task)+'.pt'
    if safe: torch.save(trajectory_set, label)
    print(f'Save Task Data! {tag}\t{task} ~ {task_simulations}')
    _b = time.time()
    print(f'Simul Time!\t{_b-_a}')
    collect_zero = (theta_set, trajectory_set)
    return collect_zero

#%%# Simulation Data Load [Local Computer]

def simul_data_load(tasks, path, tag):
    import os
    import re
    if tasks is None:
        _arcs = os.listdir(path)
        arcs = [arc for arc in _arcs if re.findall(tag+'_', arc)]
        _arcs_alp = [arc for arc in arcs if re.findall('_Theta_Set_', arc)]
        _arcs_bet = [arc for arc in arcs if re.findall('_Trajectory_Set_', arc)]
        arcs_alp = [int(re.findall('(\d+)\.pt', arc)[0]) for arc in _arcs_alp]
        arcs_bet = [int(re.findall('(\d+)\.pt', arc)[0]) for arc in _arcs_bet]
        arcs_alp.sort()
        arcs_bet.sort()
        mess = 'Oops! Something went wrong!'
        check = arcs_alp == arcs_bet
        assert check, mess
        _tasks = arcs_alp # _tasks = arcs_bet
        tasks_mini = min(_tasks)
        tasks_maxi = max(_tasks)
    elif type(tasks) is int:
        _tasks = range(tasks)
        tasks_mini = 0
        tasks_maxi = tasks - 1
    else:
        mess = "The format is invalid! The variable 'tasks' must be either 'None' or 'integer'!"
        raise RuntimeError(mess)
    print(f'Load Task Data! {tag}\t{tasks_mini} : {tasks_maxi}\tTotal {len(_tasks)}')
    for task in _tasks:
        if task == tasks_mini:
            label = path+tag+'_Theta_Set_'+str(task)+'.pt'
            theta_set = torch.load(label)
            label = path+tag+'_Trajectory_Set_'+str(task)+'.pt'
            trajectory_set = torch.load(label)
        else:
            label = path+tag+'_Theta_Set_'+str(task)+'.pt'
            _theta_set = torch.load(label)
            theta_set = torch.cat((theta_set, _theta_set), 0)
            label = path+tag+'_Trajectory_Set_'+str(task)+'.pt'
            _trajectory_set = torch.load(label)
            trajectory_set = torch.cat((trajectory_set, _trajectory_set), 0)
        print(f'Load Task Data! {tag}\t{task+1} ~ {len(_tasks)}\tIndex {task}')
    print(f'Load Task Data! {tag}\t{tasks_mini} : {tasks_maxi}\tTotal {len(_tasks)}')
    collect = (theta_set, trajectory_set)
    return collect

#%%# Section [New]


