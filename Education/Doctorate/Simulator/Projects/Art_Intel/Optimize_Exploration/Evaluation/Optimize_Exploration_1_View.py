######################################
######## Optimize Exploration ########
######################################

#%%# Catalyzer

sup_comp = False # Super Computer?
inference_prod_activate = False # Activate Inference Procedure?
data_path = 'Optimize_Exploration_1' # 'Shallow_Grid_1_N_Link'
acts = [0] # [0, 1, ...]
act = max(acts)
_observe = 1 # {0, 1, ...} # L ~ {1, 2}
observe = _observe if act != 0 else None
curb = '' # {'Weak', 'Mid', 'Strong'}
restrict = {
    'Weak': {'G_EA': (750, 1500), 'N_EA': (750, 1500)},
    'Mid': {'G_EA': (0, 1000), 'N_EA': (0, 1000)},
    'Strong': {'G_EA': (0, 750), 'N_EA': (0, 750)}
}

import sys
if sup_comp:
    path = '/home/biochemsim/ramirez/mars_projects/optimize_exploration/resources/'
else:
    path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/Art_Intel/Optimize_Exploration/Resources/'
sys.path.append(path)
from BiochemStem import BiochemStem
from BiochemSimulUltimate import BiochemSimulSwift
import numpy as np
import torch
import time
import pickle
if not sup_comp:
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 250
    plt.rcParams['savefig.dpi'] = 250

#%%# Biochemical System Construction [Preparation]

def construct_stem(parameter_set, parameter_set_true, para_fun, sup_comp = False, verbose = False):
    
    if sup_comp: verbose = False # Enforce!
    
    # Transcription Regulation [Components AND Interactions]
    
    regulation_transcription = { # ([0|1] = [Up|Down]-Regulate, Positive Integer = Transcription Cooperativity)
        'N': {'N': (0, 4), 'G': (1, 4), 'FC': (0, 2)},
        'G': {'N': (1, 4), 'G': (0, 4), 'FC': (1, 2)},
        'EA': {'N': (1, 3), 'G': (0, 3)}
    }
    
    # Species [Specs] # Transmembrane
    
    _species_promoter_state = ['I', 'A'] # {'I': 'Inactive', 'A': 'Active'}
    _species_transcription = ['N', 'G', 'FC']
    _species_translation = _species_transcription.copy()
    _species_translation.extend(['EI'])
    
    # Species [Descript]
    
    _species = {
        'promoter_state': [S + '_' + _ for S in _species_translation for _ in _species_promoter_state], # Promoter State Dynamics
        'transcription': [S + '_MRNA' for S in _species_transcription], # Explicit Transcription
        'translation': _species_translation, # Explicit Translation
        'exportation': ['FM'],
        'jump_diffuse': ['FM'],
        'dimerization': ['FD'],
        'enzymatic': ['EA'],
        'phosphorylation' : ['NP']
    }
    
    # Rate Constants
    
    diffusion_coefficients = {'N': 10e-12, 'G': 10e-12, 'NP': 10e-12, 'FC': 10e-12, 'FM': 10e-12, 'EA': 10e-12} # pow(microns, 2)/seconds # pow(length, 2)/time # Protein diffusion constant
    protein_cross_sections = {'N': 10e-9, 'G': 10e-9, 'NP': 10e-9, 'FC': 10e-9, 'FM': 10e-9, 'EA': 10e-9} # Nanometers
    binding_sites = list(regulation_transcription.keys()) # Transcription Factors!
    
    _cell_radius = 10 # Micrometers
    cell_radius = 1000e-9*_cell_radius
    cell_volume = 4*np.pi*pow(cell_radius, 3)/3 # pow(meters, 3)
    
    half_activation_thresholds = {'N_N': para_fun('N_N'), 'G_G': para_fun('G_G'), 'FC_N': para_fun('FC_N'), 'G_EA': para_fun('G_EA')} # {P}_{B} = {Promoter}_{Binding_Site}
    half_repression_thresholds = {'G_N': para_fun('G_N'), 'N_G': para_fun('N_G'), 'FC_G': para_fun('FC_G'), 'N_EA': para_fun('N_EA')} # {P}_{B} = {Promoter}_{Binding_Site}
    _rates_promoter_binding = {S: 4*np.pi*protein_cross_sections[S]*diffusion_coefficients[S]/cell_volume for S in binding_sites}
    tunes = {'N_N': 10.5, 'G_N': 10.5, 'G_G': 10.5, 'N_G': 10.5, 'G_EA': 4.375, 'N_EA': 4.375, 'FC_N': 1.775, 'FC_G': 1.775}
    _rates_promoter_unbinding = {P+'_'+B: tunes[P+'_'+B]*half_activation_thresholds[P+'_'+B]*_rates_promoter_binding[B] if regulation_transcription[B][P][0] == 0 else tunes[P+'_'+B]*half_repression_thresholds[P+'_'+B]*_rates_promoter_binding[B] for B in binding_sites for P in list(regulation_transcription[B].keys())}
    
    _MRNA_lifetime = 4 # {1, ..., 8} # Hours
    MRNA_lifetime = _MRNA_lifetime*pow(60, 2) # Seconds
    MRNA_copy_number = 250
    synthesis_spontaneous = 0.2 # [0, 1] # 1 - synthesis
    synthesis = 1 - synthesis_spontaneous
    _rates_MRNA_synthesis_spontaneous = synthesis_spontaneous*MRNA_copy_number/MRNA_lifetime
    _rates_MRNA_synthesis = synthesis*MRNA_copy_number/MRNA_lifetime
    _rates_MRNA_degradation = 1/MRNA_lifetime
    
    _protein_lifetimes = {'N': 2, 'G': 2, 'NP': 1, 'FC': 2, 'FM': 2, 'FD': 100*2, 'EI': 48} # {1, ..., 8} # Hours
    protein_lifetime = {S: _protein_lifetimes.get(S)*pow(60, 2) for S in list(_protein_lifetimes.keys())} # Seconds
    protein_copy_numbers = {'N': 4, 'G': 4, 'FC': 4, 'EI': 1000} # [100|1000] Proteins # 'P' Proteins Per MRNA
    _rates_protein_synthesis = {S: protein_copy_numbers[S]/protein_lifetime[S] for S in list(protein_copy_numbers.keys())}
    _rates_protein_degradation = {S: 1/protein_lifetime[S] for S in list(protein_lifetime.keys())}
    
    # Reactions [template = {'exes': [], 'props': [], 'deltas': [{}], 'rates': {}, 'initial_state': {}}]
    
    promoter_binding = { # {P = Promoter}_{B = Binding Site}_{C = Cooperativity}
        'exes': [f'{B} + {P}_{B}_{C} -> {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])],
        'props': [f'{B} * {P}_{B}_{C} * kb_{P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])],
        'deltas': [{B: -1, f'{P}_{B}_{C}': -1, f'{P}_{B}_{C+1}': 1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': 1 if C+1 == regulation_transcription[B][P][1] else 0} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])],
        'rates': {f'kb_{P}_{B}_{C+1}': _rates_promoter_binding[B] for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])},
        'initial_state': {f'{P}_{B}_{C}': 0 if C != 0 else 1 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]+1)}
    }
    
    promoter_binding_pho = {
        'exes': [f'{B}P + {P}_{B}_{C} -> {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'],
        'props': [f'{B}P * {P}_{B}_{C} * kb_{P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'],
        'deltas': [{f'{B}P': -1, f'{P}_{B}_{C}': -1, f'{P}_{B}_{C+1}': 1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': 1 if C+1 == regulation_transcription[B][P][1] else 0, f'{P}_{B}P_{C+1}': 1} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'],
        'rates': {f'kb_{P}_{B}_{C+1}': 1*_rates_promoter_binding[B] for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'},
        'initial_state': {f'{P}_{B}P_{C+1}': 0 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'}
    }
    
    pub = 5 # Promoter Unbinding [Coefficient]
    promoter_unbinding = { # Careful! Only auto-activation valid: zero unbinding rate when it occupies C sites!
        'exes': [f'{P}_{B}_{C+1} -> {P}_{B}_{C} + {B}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)],
        'props': [f'ku_{P}_{B}_{C} * {P}_{B}_{C+1} * (1 - {P}_{B}P_{C+1})' if B == 'N' else f'ku_{P}_{B}_{C} * {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)],
        'deltas': [{B: 1, f'{P}_{B}_{C}': 1, f'{P}_{B}_{C+1}': -1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': -1 if C+1 == regulation_transcription[B][P][1] else 0} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)],
        'rates': {f'ku_{P}_{B}_{C}': _rates_promoter_unbinding[P+'_'+B]/pow(pub, C) for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)},
        # 'rates': {f'ku_{P}_{B}_{C}': 0 if P == B and C+1 == 4 else _rates_promoter_unbinding[P+'_'+B]/pow(pub, C) for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)},
        'initial_state': {f'{P}_{B}_{C}': 0 if C != 0 else 1 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1], -1, -1)}
    }
    
    pub = 5 # Promoter Unbinding [Coefficient]
    promoter_unbinding_pho = { # Careful! Only auto-activation valid: zero unbinding rate when it occupies C sites!
        'exes': [f'{P}_{B}_{C+1} -> {P}_{B}_{C} + {B}P' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'],
        'props': [f'ku_{P}_{B}_{C} * {P}_{B}_{C+1} * {P}_{B}P_{C+1}' if B == 'N' else f'ku_{P}_{B}_{C} * {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'],
        'deltas': [{f'{B}P': 1, f'{P}_{B}_{C}': 1, f'{P}_{B}_{C+1}': -1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': -1 if C+1 == regulation_transcription[B][P][1] else 0, f'{P}_{B}P_{C+1}': -1} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'],
        'rates': {f'ku_{P}_{B}_{C}': _rates_promoter_unbinding[P+'_'+B]/pow(pub, C) for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'},
        # 'rates': {f'ku_{P}_{B}_{C}': 0 if P == B and C+1 == 4 else _rates_promoter_unbinding[P+'_'+B]/pow(pub, C) for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'},
        'initial_state': {f'{P}_{B}P_{C}': 0 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1], 0, -1) if B == 'N'}
    }
    
    MRNA_synthesis_spontaneous = {
        'exes': [f'0 -> {S}' for S in _species['transcription'] if S != 'FC_MRNA'],
        'props': [f'(1 - np.sign({S}_I)) * kss_{S}_MRNA' for S in _species_transcription if S != 'FC'],
        'deltas': [{S: 1} for S in _species['transcription'] if S != 'FC_MRNA'],
        'rates': {f'kss_{S}': _rates_MRNA_synthesis_spontaneous for S in _species['transcription'] if S != 'FC_MRNA'},
        'initial_state': {S: 0 for S in _species['transcription']}
    }
    
    MRNA_synthesis = {
        'exes': [f'{S}_A -> {S}_MRNA' for S in _species_transcription],
        'props': [f'np.sign({S}_A) * (1 - np.sign({S}_I)) * ks_{S}_MRNA' for S in _species_transcription],
        'deltas': [{S: 1} for S in _species['transcription']],
        'rates': {f'ks_{S}': _rates_MRNA_synthesis for S in _species['transcription']},
        'initial_state': {f'{S}_{_}': 0 for S in _species_transcription for _ in _species_promoter_state}
    }
    
    MRNA_degradation = {
        'exes': [f'{S} -> 0' for S in _species['transcription']],
        'props': [f'{S} * kd_{S}' for S in _species['transcription']],
        'deltas': [{S: -1} for S in _species['transcription']],
        'rates': {f'kd_{S}': _rates_MRNA_degradation for S in _species['transcription']},
        'initial_state': {S: 0 for S in _species['transcription']}
    }
    
    protein_synthesis = {
        'exes': [f'{S}_MRNA -> {S}' if S in _species_transcription else f'{S}_A -> {S}' for S in _species['translation']],
        'props': [f'{S}_MRNA * ks_{S}' if S in _species_transcription else f'{S}_A * (1 - {S}_I) * ks_{S}' for S in _species['translation']],
        'deltas': [{S: 1} for S in _species['translation']],
        'rates': {f'ks_{S}': _rates_protein_synthesis[S] for S in _species['translation']},
        'initial_state': {f'{S}_{_}': 0 if S in _species_transcription or _ == 'I' else 1 for S in _species['translation'] for _ in _species_promoter_state}
    }
    
    protein_degradation = {
        'exes': [f'{S} -> 0' for S in _species['translation']],
        'props': [f'{S} * kd_{S}' for S in _species['translation']],
        'deltas': [{S: -1} for S in _species['translation']],
        'rates': {f'kd_{S}': _rates_protein_degradation[S] for S in _species['translation']},
        'initial_state': {S: 0 for S in _species['translation']}
    }
    
    # Autocrine Signaling := {ke_F_CM} # Paracrine Signaling := {kjd_F_CM} # IN ~ Intrinsic (Start) # EX ~ Extrinsic (Final)
    
    ksig_C = 0.75*10*_rates_promoter_binding['EA']
    ksig_M = 0.75*10*_rates_promoter_binding['EA']
    
    ke_F_CM = ksig_C # (Cytoplasm)(Membrane) # Auto
    kjd_F_CM = ksig_C # (Cytoplasm)(Membrane) # Para
    # ksig_C = ke_F_CM + kjd_F_CM
    kjd_F_MM = ksig_M # (Membrane)(Membrane)
    # ksig_M = kjd_F_MM
    
    exportation = {
        'exes': ['FC -> FM'],
        'props': ['FC * ke_F_CM'],
        'deltas': [{'FC': -1, 'FM': 1}],
        'rates': {'ke_F_CM': ke_F_CM}, # + Fast != + Short # (1|10)*Promoter_Binding
        'initial_state': {'FC': 0, 'FM': 0}
    }
    
    jump_diffuse = { # Two-Step Process? # V ~ Void
        'exes': ['FC_IN -> FM_EX', 'FM_IN -> FM_EX'],
        'props': ['FC * kjd_F_CM', 'FM * kjd_F_MM'],
        'deltas': [{'FC': -1}, {'FM': -1}],
        'jump_diffuse_deltas': [{'FM': 1}, {'FM': 1}],
        'rates': {'kjd_F_CM': kjd_F_CM, 'kjd_F_MM': kjd_F_MM}, # + Fast != + Short # (1|10)*Promoter_Binding
        'initial_state': {'FC': 0, 'FM': 0}
    }
    
    dimerization = { # FD has no direct degradation, but we put it at the end!
        'exes': ['FM + FM -> FD', 'FD -> FM + FM'],
        'props': ['FM * (FM - 1) * kdf_FD', 'FD * kdb_FD'],
        'deltas': [{'FM': -2, 'FD': 1}, {'FM': 2, 'FD': -1}],
        'rates': {'kdf_FD': 1*0.5*2*_rates_promoter_binding['EA']/30, 'kdb_FD': 2*10/(2*pow(60, 2))},
        'initial_state': {'FD': 0}
    }
    
    enzymatic = { # Reverse # [Forward | Backward]
        'exes': ['FD + EI -> FD + EA', 'EA -> EI'],
        'props': ['FD * EI * kef_EA', 'EA * keb_EA'],
        'deltas': [{'EI': -1, 'EA': 1}, {'EI': 1, 'EA': -1}],
        'rates': {'kef_EA': 1/para_fun('tau_ef_EA'), 'keb_EA': 1/para_fun('tau_eb_EA')}, # 2*Promoter_Binding/10
        'initial_state': {'EI': 0, 'EA': 0}
    }
    
    phosphorylation = {
        'exes': ['EA + N -> EA + NP', 'NP -> N'],
        'props': ['EA * N * kpf_NP', 'NP * kpb_NP'],
        'deltas': [{'N': -1, 'NP': 1}, {'N': 1, 'NP': -1}],
        'rates': {'kpf_NP': 1/para_fun('tau_pf_NP'), 'kpb_NP': 1/para_fun('tau_pb_NP')}, # (1|10)*Promoter_Binding
        'initial_state': {'NP': 0}
    }
    
    _species_degradation = ['NP', 'FM', 'FD', 'EA']
    degradation = {
        'exes': [f'{S} -> 0' for S in _species_degradation],
        'props': [f'{S} * kd_{S}' for S in _species_degradation],
        'deltas': [{S: -1} for S in _species_degradation],
        'rates': {f'kd_{S}': _rates_protein_degradation['EI'] if S == 'EA' else _rates_protein_degradation[S] for S in _species_degradation},
        'initial_state': {S: 0 if S in {'EA'} else 0 for S in _species_degradation}
    }
    
    # Assemble!
    
    flags = ['promoter_binding', 'promoter_binding_pho', 'promoter_unbinding', 'promoter_unbinding_pho', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'exportation', 'jump_diffuse', 'dimerization', 'enzymatic', 'phosphorylation', 'degradation']
    initial_state = {}
    rates = {}
    for flag in flags:
        exec(f"initial_state.update({flag}['initial_state'])")
        exec(f"rates.update({flag}['rates'])")
    
    stem = BiochemStem(initial_state, rates)
    
    # flags = ['promoter_binding', 'promoter_binding_pho', 'promoter_unbinding', 'promoter_unbinding_pho', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'exportation', 'jump_diffuse', 'dimerization', 'enzymatic', 'phosphorylation', 'degradation']
    for flag in flags:
        if verbose: print(flag)
        indices = eval(f"range(len({flag}['exes']))")
        for index in indices:
            if verbose: print(index)
            name = eval(f"{flag}['exes'][{index}]")
            prop_fun = eval(f"{flag}['props'][{index}]")
            delta = eval(f"{flag}['deltas'][{index}]")
            if flag == 'jump_diffuse':
                jump_diffuse_delta = eval(f"{flag}['jump_diffuse_deltas'][{index}]") # jump_diffuse_delta = None
                # print("\n@@@@@@@@\n\tCareful! No 'jump_diffuse_delta' available!\n@@@@@@@@\n")
            else:
                jump_diffuse_delta = None
            stem.add_reaction(name, prop_fun, delta, verbose = False, jump_diffuse = jump_diffuse_delta)
    
    stem.assemble()
    
    return stem

#%%# Simulation [Preparation]

from Utilities_Shallow_Grid import instate_state_tor, instate_rate_mat
from Utilities_Shallow_Grid import interpolator
from Utilities_Shallow_Grid import plotful
from Cell_Space_Shallow_Grid import cell_placement, cell_distance, cell_neighborhood
from Cell_Space_Shallow_Grid import make_initial_pattern, make_rho_mat
from Utilities_Optimize_Exploration import make_paras
from Utilities_Shallow_Grid import make_para_fun, retrieve_paras
from Utilities_Shallow_Grid import make_simulator_ready

para_set_raw = {
    'N_N': (100, (0, 1000), 1, None), 'G_G': (200, (0, 1000), 1, None), 'FC_N': (500, (0, 1000), 1, None), 'G_EA': (750, restrict[curb]['G_EA'], 1, None),
    'G_N': (50, (0, 1000), 1, None), 'N_G': (400, (0, 1000), 1, None), 'FC_G': (500, (0, 1000), 1, None), 'N_EA': (750, restrict[curb]['N_EA'], 1, None),
    'MRNA': (50, (0, 250), 1, None), 'PRO': (200, (0, 1000), 1, None),
    'td_FC': (7200, (300, 28800), 1, None), 'td_FM': (7200, (300, 28800), 1, None),
    'tau_C': (450, (30, 5*900), 1, None), 'tau_M': (450, (30, 2*2100), 1, None),
    'tau_ef_EA': (17100, (10*30, 43200), 1, None), 'tau_eb_EA': (300, (30, 43200), 1, None), 'tau_pf_NP': (1710, (10*30, 43200), 1, None), 'tau_pb_NP': (171, (30, 43200), 1, None),
    'chi_auto': (0.5, (0, 1), 0, None)
}
para_set_mode = 0 # {0, 1} # {'No Remap', 'Remap: [A, B] ---->>>> [0, 1]'}
para_set, para_set_true = make_paras(para_set_raw = para_set_raw, para_set_mode = para_set_mode, verbose = True)

#%%# Simulation [Function]

def simulator(parameter_set, parameter_set_true, parameter_set_mode, parameter_set_rules = None, species = ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'EI', 'EA'], steps = 10000, cells = 3, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.25, cell_layers = 2, layer_cells = 1, faces = 2*3, seed = None, sup_comp = False, verbose = False):
    
    assert cells == cell_layers * layer_cells, f"We have an ill-defined number of 'cells'!\n\t{cells} ~ {cell_layers * layer_cells}"
    
    if sup_comp: verbose = False # Enforce!
    
    for rule in parameter_set_rules:
        
        para_fun = make_para_fun(parameter_set, parameter_set_true, parameter_set_mode)
        stem = construct_stem(parameter_set, parameter_set_true, para_fun, sup_comp, verbose = False)
        pat = {'N_MRNA': (para_fun('MRNA'), 0.5), 'G_MRNA': (para_fun('MRNA'), 0.5), 'N': (para_fun('PRO'), 0.5), 'G': (para_fun('PRO'), 0.5)}
        pat_mode = 'Fish_Bind'
        initial_pattern = make_initial_pattern(pat, pat_mode, verbose, species = list(stem.assembly['species'].values()), cells = cells, seed = seed)
        
        cell_location_mat, cell_size_mat = cell_placement(cell_layers, layer_cells, verbose)
        cell_hood_dit, dot_mat = cell_distance(cell_location_mat, verbose)
        cell_hoods = cell_neighborhood(cell_hood_dit, dot_mat, cell_layers, layer_cells, verbose)
        comm_classes_portrait = { # Jump-Diffuse Reactions
            0: (['FC_IN -> FM_EX', 'FM_IN -> FM_EX'], cell_hoods[0])
        }
        
        state_tor_PRE = {'N': 0, 'G': 0, 'N_MRNA': 0, 'G_MRNA': 0, 'NP': 0, 'FC': 0, 'FC_MRNA': 0, 'FM': 0, 'FD': 0, 'EI': 1000, 'EA': 0}
        state_tor_EPI = {'N': 0, 'G': 0, 'N_MRNA': 0, 'G_MRNA': 0, 'NP': 0, 'FC': 0, 'FC_MRNA': 0, 'FM': 0, 'FD': 0, 'EI': 1000, 'EA': 0}
        state_tors = {'state_tor_PRE': state_tor_PRE, 'state_tor_EPI': state_tor_EPI}
        state_tor = instate_state_tor(stem, cell_layers, layer_cells, state_tors, pat_mode, initial_pattern, blank = False)
        
        rates_exclude = list() # No Void (Lumen AND Sink_Up AND Sink_Do)
        rho_mat = make_rho_mat(cell_hood_dit, faces, cell_layers, layer_cells, verbose = False)
        kd_FC = 1/para_fun('td_FC')
        kd_FM = 1/para_fun('td_FM')
        ksig_C = rule*1/para_fun('tau_C')
        ksig_M = rule*1/para_fun('tau_M')
        chi_auto = para_fun('chi_auto')
        rate_mat = instate_rate_mat(stem, cells, parameter_set, parameter_set_true, para_fun, rates_exclude, rho_mat, blank = False, kd_FC = kd_FC, kd_FM = kd_FM, ksig_C = ksig_C, ksig_M = ksig_M, chi_auto = chi_auto)
        
        instate = {'state_tor': state_tor, 'rate_mat': rate_mat}
        simul = BiochemSimulSwift(stem, instate, steps, cells, species, seed, verbose = False)
        jump_diffuse_tor = simul.jump_diffuse_assemble(comm_classes_portrait)
        epoch_halt_tup = (time_maxi, time_unit) # Stopping Time!
        if sup_comp or verbose: _a = time.time()
        simul.meth_direct(jump_diffuse_tor, epoch_halt_tup)
        if sup_comp or verbose: _b = time.time()
        if sup_comp or verbose: print(f'Simul\nCells = {simul.cells}\tSteps = {simul.steps}\t', _b-_a)
        
        data_inter = interpolator(simul, species, time_mini, time_maxi, time_unit, time_delta, kind = 0, sup_comp = sup_comp, verbose = verbose, err = False, fill_value = 'extrapolate')
        if verbose: plotful(data_inter, species, time_unit)
        _trajectory_set = data_inter[1].flatten()
        rule_index = parameter_set_rules.index(rule)
        if rule_index == 0:
            trajectory_set = _trajectory_set
        else:
            trajectory_set = np.concatenate((trajectory_set, _trajectory_set), 0)
    
    return trajectory_set

#%%# Simulation [Arguments]

parameter_set = para_set
parameter_set_true = para_set_true
parameter_set_mode = para_set_mode
parameter_set_rules = [0, 1] # None
species = ['N', 'G', 'NP'] # ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'EI', 'EA']
steps = 7500000 if sup_comp else 2500000
cells = 5 * 5 # 'Late ICM' Plus No 'Void (Lumen /\ Sink)'
time_mini = 0
time_maxi = 48
time_unit = 'Hours'
time_delta = 0.25
cell_layers = 5
layer_cells = 5
faces = 2*3
seed = None # None
# sup_comp = False # True
verbose = True

arguments = (parameter_set, parameter_set_true, parameter_set_mode, parameter_set_rules, species, steps, cells, time_mini, time_maxi, time_unit, time_delta, cell_layers, layer_cells, faces, seed, sup_comp, verbose)

argument_values = arguments[1:len(arguments)]
simulator_ready = make_simulator_ready(simulator, argument_values)

#%%# Simulation [Local Computer Test]

if not sup_comp:
    trajectory_set = simulator(*arguments)

#%%# Inference Procedure [Preparation]

from Utilities_Shallow_Grid import make_prior
from Utilities_Optimize_Exploration import make_prior_mixer
from sbi.inference import prepare_for_sbi

prior = make_prior(para_set_true, para_set_mode, para_set_sieve = None, verbose = not sup_comp)
prior_mixer = make_prior_mixer(para_set_true, para_set_mode, para_set_sieve = None, verbose = not sup_comp)
simulator_ready, prior = prepare_for_sbi(simulator_ready, prior)
simulator_ready, prior_mixer = prepare_for_sbi(simulator_ready, prior_mixer)

#%%# Simulation /\ Inference Procedure!

tag = f'Act_{act}_Observe_{observe}_{curb.capitalize()}'
if sup_comp:
    path = f'/scratch/biochemsim/ramirez/mars_projects/optimize_exploration/data_bank/{data_path}/'
else:
    path = f'/media/mars-fias/MARS/MARS_Data_Bank/World/Optimize_Exploration/{data_path}/'

#%%# Simulation Procedure: Optimize Exploration!

from Utilities_Optimize_Exploration import instate_para_set, move_para_set
from Utilities_Optimize_Exploration import make_synopsis, refresh_synopsis, save_synopsis
from sbi.simulators.simutils import simulate_in_batches
from ObjectiveFunL1L2 import ObjectiveFunPortionRule

if not inference_prod_activate:
    
    safe = False
    verbose = False
    show = False
    if sup_comp: # Super Computer
        task_pin = int(sys.argv[1])
        tasks = 1
        task_explorations = 1000 # task_simulations
    else: # Local Computer
        task_pin = 0
        tasks = 2
        task_explorations = 10 # task_simulations
    simul_trials = 1
    if restrict[curb]: # 'Countable'
        proposal = prior_mixer
        _prior_mixer = prior_mixer
    else: # 'Uncountable'
        proposal = prior
        _prior_mixer = None
    iota = 0.025
    alpha_dit = torch.distributions.Uniform(0, 1)
    final_temp = 0.01 # (0.1|0.01)*maximum(score)
    start_temp = final_temp*task_explorations
    safe_explorations = 5 # {safe_explorations: {2, ..., task_explorations}}
    spa = ' '*8
    
    for task in range(tasks):
        print('~>'*8)
        initial_para_set = instate_para_set(proposal)
        synopsis = make_synopsis(task_explorations, para_set_true, verbose)
        for exploration in range(task_explorations):
            if exploration == 0:
                current_para_set = initial_para_set
                trajectory_set = simulate_in_batches(simulator_ready, current_para_set.repeat((simul_trials, 1)))
                objective_fun = ObjectiveFunPortionRule(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = verbose, show = show, rules = parameter_set_rules, L = _observe)
                objective_fun.apply(tau_mini = 32, tau_maxi = 48) # score_set
                current_meta_score = torch.tensor(objective_fun.appraise(perks = [2.5, 50, 97.5]))
                optimum_para_set = current_para_set
                optimum_meta_score = current_meta_score
                print(f'{task_pin}{spa}{task}{spa}{optimum_meta_score}\n{spa}{optimum_para_set.tolist()}')
                roster_varas = (current_para_set, current_meta_score, optimum_para_set, optimum_meta_score)
                synopsis = refresh_synopsis(synopsis, exploration, roster_varas, verbose)
            else:
                explore_para_set = move_para_set(current_para_set, iota, prior, _prior_mixer, verbose)
                trajectory_set = simulate_in_batches(simulator_ready, explore_para_set.repeat((simul_trials, 1)))
                objective_fun = ObjectiveFunPortionRule(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = verbose, show = show, rules = parameter_set_rules, L = _observe)
                objective_fun.apply(tau_mini = 32, tau_maxi = 48) # score_set
                explore_meta_score = torch.tensor(objective_fun.appraise(perks = [2.5, 50, 97.5]))
                delta_meta_score = explore_meta_score - current_meta_score
                alpha = alpha_dit.sample()
                temp = start_temp/(exploration+1) # exploration != 0
                eta = torch.exp(torch.tensor(-1/temp))
                kappa = torch.exp(delta_meta_score/temp)
                omega = torch.minimum(torch.tensor(1), (kappa-eta)/(1-eta))
                criterion = alpha < omega
                condition = delta_meta_score >= 0 or criterion
                if condition:
                    current_para_set = explore_para_set
                    current_meta_score = explore_meta_score
                if explore_meta_score > optimum_meta_score:
                    optimum_para_set = explore_para_set
                    optimum_meta_score = explore_meta_score
                    print(f'{task_pin}{spa}{task}{spa}{optimum_meta_score}\n{spa}{optimum_para_set.tolist()}')
                roster_varas = (explore_para_set, explore_meta_score, optimum_para_set, optimum_meta_score)
                synopsis = refresh_synopsis(synopsis, exploration, roster_varas, verbose)
            if safe and (exploration+1) % safe_explorations == 0:
                save_synopsis(synopsis, task_pin, task, path, tag)
        print('<~'*8)
    if safe:
        save_synopsis(synopsis, task_pin, task, path, tag)

#%%# Inference Procedure: Optimize Exploration!

from Utilities_Optimize_Exploration import synopsis_data_load

if inference_prod_activate:
    
    sieve = 0.0 # [0, 1]
    
    task_pins_info = None
    synopses, tasks_info = synopsis_data_load(task_pins_info, path, tag)
    _synopses_optima = torch.tensor([torch.max(synopsis[1, :, -1]) for synopsis in synopses]).sort(descending = True)
    where = _synopses_optima[0] >= sieve
    synopses_optima = _synopses_optima[0][where]
    order = _synopses_optima[1][where]
    _synopses_initial_meta_score = torch.tensor([synopsis[0, 0, -1] for synopsis in synopses])
    synopses_initial_meta_score = _synopses_initial_meta_score[order]
    task_pins = list(tasks_info.keys())[0:len(order)]
    _synopses_optima_where = torch.tensor([torch.argmax(synopsis[1, :, -1]) for synopsis in synopses])
    synopses_optima_where = _synopses_optima_where[order]
    
    plt.bar(x = task_pins, height = synopses_optima, width = 0.75, color = 'tab:blue', alpha = 0.875)
    plt.bar(x = task_pins, height = synopses_initial_meta_score, width = 0.75, color = 'tab:orange', alpha = 0.875)
    plt.hlines([0.7, 0.825, 0.95], min(task_pins)-1, max(task_pins)+1, 'tab:green', 'solid', alpha = 1)
    plt.plot(synopses_optima_where/1000, color = 'tab:red', linestyle = '', marker = '.')
    x_labels = [task_pin + 1 if task_pin == 0 or (task_pin + 1) % 10 == 0 else None for task_pin in task_pins]
    plt.xticks(task_pins, x_labels)
    y_ticks = torch.linspace(0, 1, 21)
    plt.yticks(y_ticks)
    plt.show()
