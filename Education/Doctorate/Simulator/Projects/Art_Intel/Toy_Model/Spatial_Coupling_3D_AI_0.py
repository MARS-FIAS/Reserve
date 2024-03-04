########################################
######## Spatial Coupling 3D AI ########
########################################

#%%# Catalyzer

sup_comp = False # Super Computer?

import sys
if sup_comp:
    path = '/home/biochemsim/ramirez/mars_projects/resources'
else:
    path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Resources'
sys.path.append(path)

from BiochemStem import BiochemStem
from BiochemSimul import BiochemSimulMuleRapid

# import re
import numpy as np
# import numba
# from scipy import interpolate
import torch
# import sbi
import time

if not sup_comp:
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 250
    plt.rcParams['savefig.dpi'] = 250

#%%# Biochemical System Construction [Preparation]

def construct_stem(parameter_set, parameter_set_true, para_fun, sup_comp = False, verbose = False):
    
    if sup_comp: verbose = False # Enforce!
    
    # Transcription Regulation [Components AND Interactions]
    
    regulation_transcription = { # ([0|1] = [Up|Down]-Regulate, Positive Integer = Transcription Cooperativity)
        'N': {'N': (0, 4), 'G': (1, 4), 'FI': (0, 2)},
        'G': {'N': (1, 4), 'G': (0, 4), 'FI': (1, 2)},
        'EA': {'N': (1, 3), 'G': (0, 3)}
    }
    
    # Species [Specs] # Transmembrane
    
    _species_promoter_state = ['I', 'A'] # {'I': 'Inactive', 'A': 'Active'}
    _species_transcription = ['N', 'G', 'FI']
    _species_translation = _species_transcription.copy()
    _species_translation.extend(['EI'])
    
    # Species [Descript]
    
    _species = {
        'promoter_state': [S + '_' + _ for S in _species_translation for _ in _species_promoter_state], # Promoter State Dynamics
        'transcription': [S + '_MRNA' for S in _species_transcription], # Explicit Transcription
        'translation': _species_translation, # Explicit Translation
        'exportation': ['FE'],
        'jump_diffuse': ['FE'],
        'dimerization': ['FED'],
        'enzymatic': ['EA'],
        'phosphorylation' : ['NP']
    }
    
    # Rate Constants
    
    diffusion_coefficients = { # pow(microns, 2)/seconds # pow(length, 2)/time # Protein diffusion constant
        'N': 10e-12,
        'G': 10e-12,
        'NP': 10e-12, # It does not matter!
        'FI': 10e-12,
        'FE': 10e-12,
        'EA': 10e-12
    }
    
    protein_cross_sections = { # Nanometers
        'N': 10e-9,
        'G': 10e-9,
        'NP': 10e-9, # It does not matter!
        'FI': 10e-9,
        'FE': 10e-9,
        'EA': 10e-9
    }
    
    binding_sites = list(regulation_transcription.keys()) # Transcription Factors!
    
    _cell_radius = 10 # Micrometers
    cell_radius = 1000e-9*_cell_radius
    cell_volume = 4*np.pi*pow(cell_radius, 3)/3 # pow(meters, 3)
    
    half_activation_thresholds = {'N_N': para_fun('N_N'), 'G_G': para_fun('G_G'), 'FI_N': para_fun('FI_N'), 'G_EA': para_fun('G_EA')} # {P}_{B} = {Promoter}_{Binding_Site}
    half_repression_thresholds = {'G_N': para_fun('G_N'), 'N_G': para_fun('N_G'), 'FI_G': para_fun('FI_G'), 'N_EA': para_fun('N_EA')} # {P}_{B} = {Promoter}_{Binding_Site}
    _rates_promoter_binding = {S: 4*np.pi*protein_cross_sections[S]*diffusion_coefficients[S]/cell_volume for S in binding_sites}
    tunes = {'N_N': 10.5, 'G_N': 10.5, 'G_G': 10.5, 'N_G': 10.5, 'G_EA': 4.375, 'N_EA': 4.375, 'FI_N': 1.775, 'FI_G': 1.775} # {'N_N': 10, 'N_EA': 10, 'FI_N': 10}
    _rates_promoter_unbinding = {P+'_'+B: tunes[P+'_'+B]*half_activation_thresholds[P+'_'+B]*_rates_promoter_binding[B] if regulation_transcription[B][P][0] == 0 else tunes[P+'_'+B]*half_repression_thresholds[P+'_'+B]*_rates_promoter_binding[B] for B in binding_sites for P in list(regulation_transcription[B].keys())}
    
    _MRNA_lifetime = 4 # {1, ..., 8} # Hours
    MRNA_lifetime = _MRNA_lifetime*pow(60, 2) # Seconds
    MRNA_copy_number = 250
    synthesis_spontaneous = 0.2 # [0, 1] # 1 - synthesis
    synthesis = 1 - synthesis_spontaneous
    _rates_MRNA_synthesis_spontaneous = synthesis_spontaneous*MRNA_copy_number/MRNA_lifetime
    _rates_MRNA_synthesis = synthesis*MRNA_copy_number/MRNA_lifetime
    _rates_MRNA_degradation = 1/MRNA_lifetime
    
    _protein_lifetimes = {'N': 2, 'G': 2, 'NP': 1, 'FI': 2, 'FE': 10*0.2, 'FV': 10*0.2, 'FS': 10*0.2, 'FED': 100*20*0.1, 'EI': 48} # {1, ..., 8} # Hours
    protein_lifetime = {S: _protein_lifetimes.get(S)*pow(60, 2) for S in list(_protein_lifetimes.keys())} # Seconds
    protein_copy_numbers = { # [100 | 1000] Proteins # 4 Proteins Per MRNA
        'N': 4,
        'G': 4,
        'FI': 4,
        'EI': 1000
    }
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
        'exes': [f'0 -> {S}' for S in _species['transcription'] if S != 'FI_MRNA'],
        'props': [f'(1 - np.sign({S}_I)) * kss_{S}_MRNA' for S in _species_transcription if S != 'FI'],
        'deltas': [{S: 1} for S in _species['transcription'] if S != 'FI_MRNA'],
        'rates': {f'kss_{S}': _rates_MRNA_synthesis_spontaneous for S in _species['transcription'] if S != 'FI_MRNA'},
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
    
    # Autocrine Signaling := {ke_FIE} # Paracrine Signaling := {kjd_FIE, kjd_FIV}
    
    k_sig = 0.75*10*_rates_promoter_binding['EA']
    
    ke_FIE = k_sig # Auto # ke_F(FROM)(TO)
    exportation = { # V ~ Void # effluxion
        'exes': ['FI -> FE'],
        'props': ['FI * ke_FIE'],
        'deltas': [{'FI': -1, 'FE': 1, 'FIE_AUTO': 1}],
        'rates': {'ke_FIE': ke_FIE}, # + Fast != + Short # (1|10)*Promoter_Binding
        'initial_state': {'FI': 0, 'FE': 0, 'FIE_AUTO': 0, 'FV': 0, 'FS': 0} # {'FI': 1000, 'FE': 0, 'FV': 0}
    }
    
    kjd_FIE = k_sig # Para Cell # kjd_F(FROM)(TO)
    kjd_FIV = k_sig # Para Void
    kjd_FIS = k_sig # Para Sink
    # ke_FIE + kjd_FIE + kjd_FIV + kjd_FIS = k_sig = '1'
    kjd_FE = k_sig # Cell
    kjd_FEV = k_sig # Cell Void
    kjd_FES = k_sig # Cell Sink
    # kjd_FE + kjd_FEV + kjd_FES = '1'
    kjd_FVE = k_sig # Void Cell
    kjd_FSE = k_sig # Sink Cell
    jump_diffuse = { # Two-Step Process? # V ~ Void
        'exes': ['FI_IN -> FE_OUT', 'FI_IN -> FV_OUT', 'FI_IN -> FS_OUT', 'FE_IN -> FE_OUT', 'FE_IN -> FV_OUT', 'FE_IN -> FS_OUT', 'FV_IN -> FE_OUT', 'FS_IN -> FE_OUT'],
        'props': ['FI * kjd_FIE', 'FI * kjd_FIV', 'FI * kjd_FIS', 'FE * kjd_FE', 'FE * kjd_FEV', 'FE * kjd_FES', 'FV * kjd_FVE', 'FS * kjd_FSE'],
        'deltas': [{'FI': -1, 'FE_OUT': 1}, {'FI': -1, 'FV_OUT': 1}, {'FI': -1, 'FS_OUT': 1}, {'FE': -1, 'FE_OUT': 1}, {'FE': -1, 'FV_OUT': 1}, {'FE': -1, 'FS_OUT': 1}, {'FV': -1, 'FE_OUT': 1}, {'FS': -1, 'FE_OUT': 1}],
        'jump_diffuse_deltas': [{'FE': 1, 'FI_IN': 1}, {'FV': 1, 'FI_IN': 1}, {'FS': 1, 'FI_IN': 1}, {'FE': 1, 'FE_IN': 1}, {'FV': 1, 'FE_IN': 1}, {'FS': 1, 'FE_IN': 1}, {'FE': 1, 'FV_IN': 1}, {'FE': 1, 'FS_IN': 1}],
        'rates': {'kjd_FIE': kjd_FIE, 'kjd_FIV': kjd_FIV, 'kjd_FIS': kjd_FIS, 'kjd_FE': kjd_FE, 'kjd_FEV': kjd_FEV, 'kjd_FES': kjd_FES, 'kjd_FVE': kjd_FVE, 'kjd_FSE': kjd_FSE}, # + Fast != + Short # (1|10)*Promoter_Binding
        'initial_state': {'FI_IN': 0, 'FI_OUT': 0, 'FE_IN': 0, 'FE_OUT': 0, 'FV_IN': 0, 'FV_OUT': 0, 'FS_IN': 0, 'FS_OUT': 0}
    }
    
    dimerization = { # FED has no direct degradation, but we put it at the end!
        'exes': ['FE + FE -> FED', 'FED -> FE + FE'],
        'props': ['FE * (FE - 1) * kdf_FED', 'FED * kdb_FED'],
        'deltas': [{'FE': -2, 'FED': 1}, {'FE': 2, 'FED': -1}],
        'rates': {'kdf_FED': 1*0.5*2*_rates_promoter_binding['EA']/30, 'kdb_FED': 2*10/(2*pow(60, 2))},
        'initial_state': {'FED': 0}
    }
    
    enzymatic = { # Reverse # [Forward | Backward]
        'exes': ['FED + EI -> FED + EA', 'EA -> EI'],
        'props': ['FED * EI * kef_EA', 'EA * keb_EA'],
        'deltas': [{'EI': -1, 'EA': 1}, {'EI': 1, 'EA': -1}],
        'rates': {'kef_EA': 1*2*_rates_promoter_binding['EA']/10, 'keb_EA': 1*50*2*_rates_promoter_binding['EA']/10}, # 2*Promoter_Binding/10
        'initial_state': {'EI': 0, 'EA': 0} # {'EI': 100, 'EA': 0}
    }
    
    phosphorylation = {
        'exes': ['EA + N -> EA + NP', 'NP -> N'],
        'props': ['EA * N * kf_NP', 'NP * kb_NP'],
        'deltas': [{'N': -1, 'NP': 1}, {'N': 1, 'NP': -1}],
        'rates': {'kf_NP': 1*2*_rates_promoter_binding['EA'], 'kb_NP': 20*_rates_promoter_binding['EA']}, # (1|10)*Promoter_Binding
        'initial_state': {'NP': 0}
    }
    
    _species_degradation = ['FE', 'FV', 'FS', 'FED', 'EA', 'NP']
    degradation = {
        'exes': [f'{S} -> 0' for S in _species_degradation],
        'props': [f'{S} * kd_{S}' for S in _species_degradation],
        'deltas': [{S: -1} for S in _species_degradation],
        'rates': {f'kd_{S}': _rates_protein_degradation['EI'] if S == 'EA' else _rates_protein_degradation[S] for S in _species_degradation},
        'initial_state': {S: 0 if S in {'EA'} else 0 for S in _species_degradation}
    }
    
    ke_VS = k_sig
    void = {
        'exes': ['0 -> V', 'V -> 0', 'V -> S'],
        'props': ['ks_V', 'V * kd_V', 'V * ke_VS'],
        'deltas': [{'V': 1}, {'V': -1}, {'V': -1, 'S': 1, 'VS': 1}],
        'rates': {'ks_V': 10 / (72 * pow(60, 2)), 'kd_V': 1 / (72 * pow(60, 2)), 'ke_VS': ke_VS},
        'initial_state': {'V': 0, 'S': 0, 'VS': 0}
    }
    
    ke_SV = k_sig
    sink = {
        'exes': ['0 -> S', 'S -> 0', 'S -> V'],
        'props': ['ks_S', 'S * kd_S', 'S * ke_SV'],
        'deltas': [{'S': 1}, {'S': -1}, {'S': -1, 'V': 1, 'SV': 1}],
        'rates': {'ks_S': 10 / (72 * pow(60, 2)), 'kd_S': 1 / (72 * pow(60, 2)), 'ke_SV': ke_SV},
        'initial_state': {'S': 0, 'V': 0, 'SV': 0}
    }
    
    # None
    
    flags = ['promoter_binding', 'promoter_binding_pho', 'promoter_unbinding', 'promoter_unbinding_pho', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'exportation', 'jump_diffuse', 'dimerization', 'enzymatic', 'phosphorylation', 'degradation', 'void', 'sink']
    initial_state = {}
    rates = {}
    for flag in flags:
        exec(f"initial_state.update({flag}['initial_state'])")
        exec(f"rates.update({flag}['rates'])")
    
    stem = BiochemStem(initial_state, rates)
    
    # flags = ['promoter_binding', 'promoter_binding_pho', 'promoter_unbinding', 'promoter_unbinding_pho', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'exportation', 'jump_diffuse', 'dimerization', 'enzymatic', 'phosphorylation', 'degradation', 'void', 'sink']
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

from Utilities import instate_state_tor, instate_rate_mat
from Utilities import make_jump_diffuse_tor_simul
from Utilities import interpolator
from Utilities import plotful

from Cell_Space import cell_placement, cell_distance, cell_neighborhood
from Cell_Space import make_rho_mat

from Utilities import make_paras, make_para_fun
from Utilities import make_simulator_ready

para_set, para_set_true = make_paras(exclude = ['Initial_Pat'], verbose = True)

#%%# Simulation [Function]

def simulator(parameter_set, parameter_set_true, species = ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FED', 'EA'], steps = 10000, cells = 3, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, cell_layers = 2, layer_cells = 1, faces = 2*3, seed = None, jump_diffuse_seed = 0, sup_comp = False, verbose = False):
    
    assert cells == cell_layers * layer_cells + 1, f"We have an ill-defined number of 'cells'!\n\t{cells} ~ {cell_layers * layer_cells + 1}"
    
    if sup_comp: verbose = False # Enforce!
    
    para_fun = make_para_fun(parameter_set, parameter_set_true)
    stem = construct_stem(parameter_set, parameter_set_true, para_fun, sup_comp, verbose = False)
    
    cell_location_mat, cell_size_mat = cell_placement(cell_layers, layer_cells, verbose)
    cell_hood_dit, dot_mat = cell_distance(cell_location_mat, verbose)
    cell_hoods = cell_neighborhood(cell_hood_dit, dot_mat, cell_layers, layer_cells, verbose)
    comm_classes_portrait = { # Jump-Diffuse Reactions
        0: (['FI_IN -> FE_OUT', 'FE_IN -> FE_OUT'], cell_hoods[0]),
        1: (['FI_IN -> FV_OUT', 'FE_IN -> FV_OUT', 'FV_IN -> FE_OUT'], cell_hoods[1]),
        2: (['FI_IN -> FS_OUT', 'FE_IN -> FS_OUT', 'FS_IN -> FE_OUT'], cell_hoods[2])
    }
    
    state_tor_PRE = {'N': 0*1000, 'G': 1*1000, 'N_MRNA': 0*250, 'G_MRNA': 1*250, 'FI': 0*800, 'FI_MRNA': 0*200, 'EI': 1*1000}
    state_tor_EPI = {'N': 1*1000, 'G': 0*1000, 'N_MRNA': 1*250, 'G_MRNA': 0*250, 'FI': 1*800, 'FI_MRNA': 1*200, 'EI': 1*1000}
    state_tor_V = {'N': 0, 'G': 0, 'N_MRNA': 0, 'G_MRNA': 0, 'FI': 0, 'FI_MRNA': 0, 'EI': 0}
    state_tors = {'state_tor_PRE': state_tor_PRE, 'state_tor_EPI': state_tor_EPI, 'state_tor_V': state_tor_V}
    state_tor = instate_state_tor(stem, cell_layers, layer_cells, state_tors, blank = False)
    
    rates_exclude = ['ks_V', 'kd_V', 'ke_VS', 'ks_S', 'kd_S', 'ke_SV', 'kjd_FVE', 'kjd_FSE', 'kd_FV', 'kd_FS'] # Void/Lumen AND Sink
    rho_mat = make_rho_mat(cell_hood_dit, faces, cell_layers, layer_cells, verbose = False)
    rate_mat = instate_rate_mat(stem, cells, parameter_set, parameter_set_true, para_fun, rates_exclude, rho_mat, blank = False)
    
    instate = {'state_tor': state_tor, 'rate_mat': rate_mat}
    simul = BiochemSimulMuleRapid(stem, instate, steps, cells, seed, verbose = False)
    
    jump_diffuse_tor, simul = make_jump_diffuse_tor_simul(simul, comm_classes_portrait, jump_diffuse_seed, blank = False)
    
    epoch_halt_tup = (time_maxi, time_unit) # Stopping Time!
    
    if sup_comp or verbose: _a = time.time()
    simul.meth_direct(jump_diffuse_tor, epoch_halt_tup)
    if sup_comp or verbose: _b = time.time()
    if sup_comp or verbose: print(f'Simul\nCells = {simul.cells}\tSteps = {simul.steps}\t', _b-_a)
    
    kind = 0
    data_inter = interpolator(simul, species, time_mini, time_maxi, time_unit, time_delta, kind, sup_comp, verbose)
    if verbose: plotful(data_inter, species, time_unit)
    
    trajectory_set = data_inter[1].flatten()
    
    return trajectory_set

#%%# Simulation [Arguments]

parameter_set = para_set
parameter_set_true = para_set_true
species = ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FED', 'EA']
steps = 2500000
cells = 12 + 1 # 'Late ICM' Plus 'Cavity/Sink'
time_mini = 0
time_maxi = 48
time_unit = 'Hours'
time_delta = 0.2
cell_layers = 3
layer_cells = 4
faces = 2*3
seed = None # None
jump_diffuse_seed = 0 if not sup_comp else int(sys.argv[1])
# sup_comp = False # True
verbose = True

arguments = (parameter_set, parameter_set_true, species, steps, cells, time_mini, time_maxi, time_unit, time_delta, cell_layers, layer_cells, faces, seed, jump_diffuse_seed, sup_comp, verbose)

argument_values = arguments[1:-1]
simulator_ready = make_simulator_ready(simulator, argument_values)

#%%# Simulation [Local Computer Test]

if not sup_comp:
    trajectory_set = simulator(*arguments)

#%%# Inference Procedure [Preparation]

from sbi import utils, analysis

from sbi.inference import prepare_for_sbi, simulate_for_sbi
from sbi.inference import SNPE

from Utilities import simul_procedure_loa_comp, simul_procedure_sup_comp
from Utilities import simul_data_load
from Utilities import restructure, previewer
from Utilities import sieve_data, comb_tram_data, objective_fun_demo

para_span = np.array([0, 1]) # Parameter Range
card = para_set.shape[0] # Parameter Set Cardinality
prior = utils.BoxUniform(low = para_span[0]*torch.ones(card), high = para_span[1]*torch.ones(card))
simulator_ready, prior = prepare_for_sbi(simulator_ready, prior)

#%%# Inference Procedure: Simul Data Save! [Local Computer]

if not sup_comp:
    tasks = 2
    task_simulations = 5
    path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Resources/Alpha_Tests/Data_Bank/Spatial_Coupling_3D_AI_0/'
    tag = 'Loa_Comp'
    safe = True
    collect_last = simul_procedure_loa_comp(simulator_ready, prior, tasks, task_simulations, path, tag, safe)

#%%# Inference Procedure: Simul Data Save! [Super Computer]

if sup_comp:
    task = int(sys.argv[1])
    task_simulations = 10
    path = '/scratch/biochemsim/ramirez/mars_projects/spatial_coupling_art_intel/data_bank/Spatial_Coupling_3D_AI_0/'
    tag = 'Sup_Comp'
    safe = True
    collect_zero = simul_procedure_sup_comp(simulator_ready, prior, task, task_simulations, path, tag, safe)

#%%# Inference Procedure: Simul Data Load! [Super Computer ---->>>> Local Computer]

if not sup_comp:
    tasks = 10
    path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Resources/Alpha_Tests/Data_Bank/Spatial_Coupling_3D_AI_0/'
    tag = 'Sup_Comp' # 'Loa_Comp'
    theta_set, trajectory_set = simul_data_load(tasks, path, tag)

#%%# Inference Procedure: Simul Data Plot! [Super Computer ---->>>> Local Computer]

if not sup_comp:
    trajectories = [0, 1]
    observations = trajectory_set[trajectories]
    simulations_maxi = 10
    _ = restructure(observations, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose)

#%%# Inference Procedure: Train ANN! [Simul Data]

if not sup_comp:
    inference = SNPE(prior)
    inference = inference.append_simulations(theta_set, trajectory_set)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

#%%# Inference Procedure: Posterior Samples! [Simul Data]

if not sup_comp:
    trajectory = 0
    observation = trajectory_set[trajectory]
    mu, tasks, task_simulations = (2, 10, 1000)
    posterior_samples = posterior.sample(sample_shape = tuple([mu*tasks*task_simulations]), x = observation)
    _ = analysis.pairplot(samples = posterior_samples, labels = [rf'$\theta_{c}$' for c in range(card)], points = theta_set[trajectory], points_colors = 'r', points_offdiag = {'markersize': 4}, limits = [para_span]*card, figsize = (7.5, 7.5))
    print(posterior)
    print(f'Theta Set!\n\t{theta_set[trajectory]}')

#%%# Inference Procedure: Objective Function Demo! [Preprocessor AND Processor]

if not sup_comp:
    trajectories = np.arange(100) # [0, 1]
    simulations_maxi = 2 # Show Few Trajectories!
    data_rest = restructure(trajectory_set[trajectories], species, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, simulations_maxi = 10, sup_comp = False, verbose = False)
    previewer(data_rest, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose = False)
    species_sieve = ['N', 'G', 'NP']
    data_sieve = sieve_data(trajectory_set, species, species_sieve, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose = False)
    previewer(data_sieve[trajectories], species_sieve, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose)
    species_comb_tram_dit = {'NT': (('N', 'NP'), np.add)} # {'GP': (('G', 0.5), np.power)}
    data_comb_tram, species_comb_tram = comb_tram_data(data_sieve, species_sieve, species_comb_tram_dit, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose)
    previewer(data_comb_tram[trajectories], species_comb_tram, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose)
    species_sieve = ['NT', 'G']
    data_sieve = sieve_data(data_comb_tram, species_comb_tram, species_sieve, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose = False)
    previewer(data_sieve[trajectories], species_sieve, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp, verbose)
    data_objective = objective_fun_demo(data_sieve, species_sieve, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, verbose)

#%%# Inference Procedure: Objective Function! [Preparation]

if not sup_comp:
    from ObjectiveFun import ObjectiveFunZero, ObjectiveFunOne
    from InferenceProd import InferenceProd
    simulations_maxi = 10
    objective_fun_reference = ObjectiveFunOne(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp)
    inference_prod_reference = InferenceProd(objective_fun_reference, theta_set, prior)
    inference_prod_reference.apply()

#%%# Inference Procedure: Objective Function! [Synthetic Data]

if not sup_comp:
    trajectories = list(range(10, 15))
    posterior_sample_shape = tuple([20000])
    inference_prod_reference.examiner(trajectories, posterior_sample_shape, parameter_set_true)
    objective_fun = ObjectiveFunOne(None, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi, sup_comp)
    observation = 1*torch.ones(objective_fun_reference.data_objective.shape[2])
    inference_prod_reference.verbose = True # View Raw (Simulation) Data!
    inference_prod_reference.synthesizer(observation, posterior_sample_shape, parameter_set_true, simulator_ready, objective_fun)
