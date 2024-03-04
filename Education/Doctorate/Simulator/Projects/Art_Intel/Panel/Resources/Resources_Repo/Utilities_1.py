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

def interpolator(simul, species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, kind = 0, sup_comp = False, verbose = False):
    
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
    
    values = list(simul.stem.assembly['species'].values())
    _s = species
    s = [values.index(value) for value in _s]
    
    cells = simul.cells
    data_ix = ix
    data_iy = np.zeros((len(s), len(ix), cells))
    
    for i in range(cells):
        for j in s:
            x = simul.epoch_mat[:, i]
            y = simul.state_tor[:, j, i]
            fun = interpolate.interp1d(x = x, y = y, kind = kind)
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
    x_mini = 1*np.min(x)
    x_maxi = 1.025*np.max(x)
    
    y_mini = 1*np.min(iy)
    y_maxi = 1.025*np.max(iy)
    
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

def instate_rate_mat(stem, cells, parameter_set, parameter_set_true, para_fun, rates_exclude, rho_mat, blank = False):
    if blank:
        rate_mat = None
    else:
        k_sig = stem.rates['ke_FIE'] # 0.75*10*_rates_promoter_binding['EA'] # Intracellular -> Extracellular
        k_men = stem.rates['kjd_FIE'] # 0.75*10*_rates_promoter_binding['EA'] # Membrane
        phi_auto = 0.5 # Auto # [0, 1]
        phi_para = 1 - phi_auto # Para # [0, 1]
        psi = 1 # {Membrane, Void, Sink} # [0, 'infinity')
        psi_void = 1 # Void # [0, 'infinity')
        psi_sink = 1 # Sink # [0, 'infinity')
        psi_void_sink = 1 # From Void To Sink # [0, 'infinity')
        psi_sink_void = 1 # From Sink To Void # [0, 'infinity')
        rate_mat_V = {'kjd_FVE': psi_void*k_men, 'kjd_FSE': psi_sink*k_men, 'ke_VS': psi_void_sink*k_men, 'ke_SV': psi_sink_void*k_men}
        rate_mat_V.update({key: 0 for key in stem.rates.keys() if key not in rates_exclude})
        # ke_FIE + kjd_FIE + kjd_FIV + kjd_FIS = '1' # kjd_FE + kjd_FEV + kjd_FES = '1' # kjd_FVE = ? # kjd_FSE = ?
        for cell in range(cells):
            exec(f'_rate_mat_{cell} = stem.rates.copy()')
            if cell != cells -1:
                rho_cell, rho_void, rho_sink = rho_mat[cell]
                rate_mat_ICM = {
                    'ke_FIE': phi_auto*k_sig,
                    'kjd_FIE': rho_cell*phi_para*k_sig, 'kjd_FIV': rho_void*phi_para*k_sig, 'kjd_FIS': 1*rho_sink*phi_para*k_sig,
                    'kjd_FE': psi*rho_cell*k_men, 'kjd_FEV': psi*rho_void*k_men, 'kjd_FES': 1*psi*rho_sink*k_men,
                    'kjd_FVE': 0, 'kjd_FSE': 0
                }
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

def make_paras(I_P = (0, 1), P_B = (0, 1000), include = None, exclude = None, verbose = False):
    descript = "{'Parameter': ('Act_Value', ('Mini_Value', 'Maxi_Value'))}"
    if verbose: print(descript)
    if include is not None:
        mess = "The variable 'include' must be a valid dictionary!"
        check = type(include) is dict
        assert check, mess
        mess = 'Every dictionary key must be a string!'
        _check = [type(key) is str for key in include.keys()]
        check = all(_check)
        assert check, mess
        mess = 'Every dictionary value must be a duple!'
        _check = [type(value) is tuple and len(value) == 2 for value in include.values()]
        check = all(_check)
        assert check, mess
        mess = "Every duple must specify a valid 'true' parameter value and a valid 'duple' parameter range!"
        _check = [value[1][0] <= value[0] <= value[1][1] for value in include.values()]
        check = all(_check)
        assert check, mess
        para_set_true = include
    else:
        # I_P = (0, 1) # {I}_{P} = {Initial}_{Pattern} # Rational Numbers!
        # P_B = (0, 1000) # {P}_{B} = {Promoter}_{Binding_Site}
        para_set_true = {
            'Initial_Pat': (1, I_P), # Initial Pattern!
            'N_N': (125, P_B), 'G_G': (500, P_B), 'FI_N': (125, P_B), 'G_EA': (500, P_B), # Half-Activation Thresholds
            'G_N': (125, P_B), 'N_G': (750, P_B), 'FI_G': (125, P_B), 'N_EA': (500, P_B) # Half-Repression Thresholds
        }
        if exclude is not None:
            for key in exclude:
                para_set_true.pop(key)
    para_set = np.array([(value[0]-value[1][0])/(value[1][1]-value[1][0]) for value in para_set_true.values()])
    paras = (para_set, para_set_true)
    return paras

#%%# Retrieve Paras

def retrieve_paras(para_set, para_set_true, verbose = False):
    para_set_act = dict()
    para_keys = list(para_set_true.keys())
    for para_key in para_keys:
        alp = para_set_true[para_key][1][0]
        bet = para_set_true[para_key][1][1]
        chi = bet - alp
        para_value = chi * para_set[para_keys.index(para_key)] + alp
        para_set_act.update({para_key: para_value})
    if verbose: print(para_set_true)
    return para_set_act

#%%# Make Para Fun [Closure]

def make_para_fun(parameter_set, parameter_set_true):
    _parameter_set_true = list(parameter_set_true.keys())
    parameter_set_index = {key: _parameter_set_true.index(key) for key in _parameter_set_true}
    def para_fun(parameter_key):
        alp = parameter_set_true[parameter_key][1][0]
        bet = parameter_set_true[parameter_key][1][1]
        chi = bet - alp
        parameter_value = chi * parameter_set[parameter_set_index[parameter_key]] + alp
        return parameter_value
    return para_fun

#%%# Make Simulator Ready [Partial Object]

def make_simulator_ready(simulator, argument_values):
    import inspect
    from functools import partial
    argument_keys = tuple(inspect.signature(simulator).parameters.keys())[1:-1]
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
    for task in range(tasks):
        if task == 0:
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
        print(f'Load Task Data! {tag}\t{task+1} ~ {tasks}\tIndex {task}')
    collect = (theta_set, trajectory_set)
    return collect

#%%# Sieve Data

def sieve_data(data, species, species_sieve, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, sup_comp = False, verbose = False):
    
    check = len(data.shape) # (simulations, len(species), len(x), cells)
    if check != 4:
        data = restructure(data, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose)
    
    check = [s in species for s in species_sieve]
    mess = f"The list 'species_sieve' is not consistent with the 'species' list!\n\t{np.array(species_sieve)[np.invert(check)]}"
    assert all(check), mess
    
    indices = [species.index(s) for s in species_sieve]
    data_sieve = np.copy(data[:, indices, :, :])
    
    return data_sieve

#%%# Combine OR Transform Data

def comb_tram_data(data, species, species_comb_tram_dit, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, sup_comp = False, verbose = False):
    
    check = len(data.shape) # (simulations, len(species), len(x), cells)
    if check != 4:
        data = restructure(data, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose)
    
    specs = list(species_comb_tram_dit.keys())
    check = [spec in species for spec in specs]
    mess = f'We must provide a new name for each novel species!\n\t{specs}'
    assert not any(check), mess
    
    species_comb_tram = species.copy()
    species_comb_tram.extend(specs)
    data_comb_tram = np.copy(data)
    
    for spec in specs:
        elements = species_comb_tram_dit.get(spec)
        spas = elements[0]
        check_alp = [isinstance(spa, str) for spa in spas]
        mess = 'At least one of the arguments must be a valid species name or string!'
        assert any(check_alp), mess
        check_bet = np.array(spas)[check_alp]
        check = [s in species for s in check_bet]
        mess = f'Invalid species!\n\t{check_bet}'
        assert all(check), mess
        fun = elements[1]
        check = isinstance(fun, np.ufunc)
        mess = "The 'comb_tram' function must be a 'NUMPY' universal function"
        assert check, mess
        arguments = [data_comb_tram[:, [species.index(spa)], :, :] if spa in species else spa for spa in spas]
        temp = fun(*arguments)
        data_comb_tram = np.append(data_comb_tram, temp, 1)
    
    comb_tram = (data_comb_tram, species_comb_tram)
    
    return comb_tram

#%%# Objective Function [Demo]

def objective_fun_demo(data, species = ['NT', 'G'], time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, verbose = False):
    
    data_objective = np.full((data.shape[0], 1, data.shape[2], data.shape[3]), np.nan)
    simulations = data.shape[0]
    cells = data.shape[-1] # data.shape[3]
    x = np.arange(time_mini, time_maxi+1, time_delta)
    
    NT_E = np.sum(data[:, species.index('NT'), :, :], 2) # Embryo
    NT_C = np.copy(data[:, species.index('NT'), :, :]) # Cell
    G_E = np.sum(data[:, species.index('G'), :, :], 2) # Embryo
    G_C = np.copy(data[:, species.index('G'), :, :]) # Cell
    a = NT_C/(NT_C+G_C)
    b = G_C/(NT_C+G_C)
    c = NT_E/(NT_E+G_E)
    d = G_E/(NT_E+G_E)
    for cell in range(4):
        data_objective[:, 0, :, cell] = d*b[..., cell]/(1-(d*a[..., cell]+c*b[..., cell]))
    for cell in range(4, cells-1):
        data_objective[:, 0, :, cell] = c*a[..., cell]/(1-(d*a[..., cell]+c*b[..., cell]))
    
    for simulation in range(simulations):
        if simulation >= simulations_maxi:
            break
        for cell in range(cells-1):
            tit = f'Simulation ~ Cell\n{simulation} ~ {cell}'
            y = data[simulation, :, :, cell].T
            plt.plot(x, y, drawstyle = 'steps-post', linestyle = '-')
            plt.title(tit)
            plt.xlabel(time_unit)
            plt.ylabel('Copy Number')
            plt.legend(species)
            plt.grid(linestyle = '--')
            plt.show()
            color = 'tab:green' if cell >= 4 else 'tab:red'
            plt.plot(x, data_objective[simulation, 0, :, cell], color = color)
            plt.title(tit)
            plt.xlabel(time_unit)
            plt.ylabel('Score')
            plt.xlim(time_mini, time_maxi)
            plt.ylim(-0.1, 1.1)
            plt.grid(linestyle = '--')
            plt.show()
    
    return data_objective

#%%# Section [New]


