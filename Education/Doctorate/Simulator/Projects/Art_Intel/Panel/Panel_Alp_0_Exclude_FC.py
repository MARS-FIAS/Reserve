###########################
######## Panel Alp ########
###########################

#%%# Catalyzer

sup_comp = False # Super Computer?
inference_prod_activate = False # Activate Inference Procedure?
data_path = 'Panel_Alp_0_Exclude_FC' # 'Spatial_Coupling_3D_AI_Earth_0_N_G_FL'
acts = [0] # [0, 1, ...]
act = max(acts)
_observe = 0 # {0, 1, ...}
observe = _observe if act != 0 else None
curb = (5, 3) # {0: (5, 3), 1: (10, 4, 2), 2: (15, 5, 3, 1)}
_curb = 'Cellulate_'+'_'.join(map(str, curb))

import sys
if sup_comp:
    path = '/home/biochemsim/ramirez/mars_projects/panel/resources'
else:
    path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/Art_Intel/Panel/Resources'
sys.path.append(path)
from BiochemStem import BiochemStem
from BiochemSimul import BiochemSimulMuleRapid
import numpy as np
import torch
import time
import pickle
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
    
    _protein_lifetimes = {'N': 2, 'G': 2, 'NP': 1, 'FC': 2, 'FM': 2, 'FL': 2, 'FS_UP': 2, 'FS_DO': 2, 'FD': 100*2, 'EI': 48} # {1, ..., 8} # Hours
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
    
    # Autocrine Signaling := {ke_F_CM} # Paracrine Signaling := {kjd_F_CM, kjd_F_CV} # IN ~ Intrinsic (Start) # EX ~ Extrinsic (Final)
    
    ksig_C = 0.75*10*_rates_promoter_binding['EA']
    ksig_M = 0.75*10*_rates_promoter_binding['EA']
    ksig_L = 0.75*10*_rates_promoter_binding['EA']
    ksig_S_UP = 0.75*10*_rates_promoter_binding['EA']
    ksig_S_DO = 0.75*10*_rates_promoter_binding['EA']
    
    ke_F_CM = ksig_C # (Cytoplasm)(Membrane) # Auto
    kjd_F_CM = ksig_C # (Cytoplasm)(Membrane) # Para
    kjd_F_CL = ksig_C # (Cytoplasm)(Lumen) # Para
    kjd_F_CS_UP = ksig_C # (Cytoplasm)(Sink_Up) # Para
    kjd_F_CS_DO = 0 # (Cytoplasm)(Sink_Do) # Para
    # ksig_C = ke_F_CM + kjd_F_CM + kjd_F_CL + kjd_F_CS_UP + kjd_F_CS_DO
    kjd_F_MM = ksig_M # (Membrane)(Membrane)
    kjd_F_ML = ksig_M # (Membrane)(Lumen)
    kjd_F_MS_UP = ksig_M # (Membrane)(Sink_Up)
    kjd_F_MS_DO = 0 # (Membrane)(Sink_Do)
    # ksig_M = kjd_F_MM + kjd_F_ML + kjd_F_MS_UP + kjd_F_MS_DO
    kjd_F_LM = ksig_L # (Lumen)(Membrane)
    ke_F_LS_UP = 0 # (Lumen)(Sink_Up)
    ke_F_LS_DO = ksig_L # (Lumen)(Sink_Do)
    # ksig_L = kjd_F_LM + ke_F_LS_UP + ke_F_LS_DO
    kjd_F_SM_UP = ksig_S_UP # (Sink_Up)(Membrane)
    ke_F_SL_UP = 0 # (Sink_Up)(Lumen)
    ke_FS_UP_DO = ksig_S_UP # (Sink_Up)(Sink_Do)
    # ksig_S_UP = kjd_F_SM_UP + ke_F_SL_UP + ke_FS_UP_DO
    kjd_F_SM_DO = 0 # (Sink_Do)(Membrane)
    ke_F_SL_DO = ksig_S_DO # (Sink_Do)(Lumen)
    ke_FS_DO_UP = ksig_S_DO # (Sink_Do)(Sink_Up)
    # ksig_S_DO = kjd_F_SM_DO + ke_F_SL_DO + ke_FS_DO_UP
    
    exportation = {
        'exes': ['FC -> FM'],
        'props': ['FC * ke_F_CM'],
        'deltas': [{'FC': -1, 'FM': 1, 'F_CM_AUTO': 1}],
        'rates': {'ke_F_CM': ke_F_CM}, # + Fast != + Short # (1|10)*Promoter_Binding
        'initial_state': {'FC': 0, 'FM': 0, 'F_CM_AUTO': 0}
    }
    
    jump_diffuse = { # Two-Step Process? # V ~ Void
        'exes': ['FC_IN -> FM_EX', 'FC_IN -> FL_EX', 'FC_IN -> FS_UP_EX', 'FC_IN -> FS_DO_EX', 'FM_IN -> FM_EX', 'FM_IN -> FL_EX', 'FM_IN -> FS_UP_EX', 'FM_IN -> FS_DO_EX', 'FL_IN -> FM_EX', 'FS_UP_IN -> FM_EX', 'FS_DO_IN -> FM_EX'],
        'props': ['FC * kjd_F_CM', 'FC * kjd_F_CL', 'FC * kjd_F_CS_UP', 'FC * kjd_F_CS_DO', 'FM * kjd_F_MM', 'FM * kjd_F_ML', 'FM * kjd_F_MS_UP', 'FM * kjd_F_MS_DO', 'FL * kjd_F_LM', 'FS_UP * kjd_F_SM_UP', 'FS_DO * kjd_F_SM_DO'],
        'deltas': [{'FC': -1, 'FM_EX': 1}, {'FC': -1, 'FL_EX': 1}, {'FC': -1, 'FS_UP_EX': 1}, {'FC': -1, 'FS_DO_EX': 1}, {'FM': -1, 'FM_EX': 1}, {'FM': -1, 'FL_EX': 1}, {'FM': -1, 'FS_UP_EX': 1}, {'FM': -1, 'FS_DO_EX': 1}, {'FL': -1, 'FM_EX': 1}, {'FS_UP': -1, 'FM_EX': 1}, {'FS_DO': -1, 'FM_EX': 1}],
        'jump_diffuse_deltas': [{'FM': 1, 'FC_IN': 1}, {'FL': 1, 'FC_IN': 1}, {'FS_UP': 1, 'FC_IN': 1}, {'FS_DO': 1, 'FC_IN': 1}, {'FM': 1, 'FM_IN': 1}, {'FL': 1, 'FM_IN': 1}, {'FS_UP': 1, 'FM_IN': 1}, {'FS_DO': 1, 'FM_IN': 1}, {'FM': 1, 'FL_IN': 1}, {'FM': 1, 'FS_UP_IN': 1}, {'FM': 1, 'FS_DO_IN': 1}],
        'rates': {'kjd_F_CM': kjd_F_CM, 'kjd_F_CL': kjd_F_CL, 'kjd_F_CS_UP': kjd_F_CS_UP, 'kjd_F_CS_DO': kjd_F_CS_DO, 'kjd_F_MM': kjd_F_MM, 'kjd_F_ML': kjd_F_ML, 'kjd_F_MS_UP': kjd_F_MS_UP, 'kjd_F_MS_DO': kjd_F_MS_DO, 'kjd_F_LM': kjd_F_LM, 'kjd_F_SM_UP': kjd_F_SM_UP, 'kjd_F_SM_DO': kjd_F_SM_DO}, # + Fast != + Short # (1|10)*Promoter_Binding
        'initial_state': {'FC_IN': 0, 'FC_EX': 0, 'FM_IN': 0, 'FM_EX': 0, 'FL_IN': 0, 'FL_EX': 0, 'FS_UP_IN': 0, 'FS_UP_EX': 0, 'FS_DO_IN': 0, 'FS_DO_EX': 0}
    }
    
    void = { # V ~ Void # L ~ Lumen # S ~ Sink # Effluxion
        'exes': ['0 -> V', 'V -> 0', 'FL -> FS_UP', 'FL -> FS_DO', 'FS_UP -> FL', 'FS_UP -> FS_DO', 'FS_DO -> FL', 'FS_DO -> FS_UP'],
        'props': ['ks_V', 'V * kd_V', 'FL * ke_F_LS_UP', 'FL * ke_F_LS_DO', 'FS_UP * ke_F_SL_UP', 'FS_UP * ke_FS_UP_DO', 'FS_DO * ke_F_SL_DO', 'FS_DO * ke_FS_DO_UP'],
        'deltas': [{'V': 1}, {'V': -1}, {'FL': -1, 'FS_UP': 1}, {'FL': -1, 'FS_DO': 1}, {'FS_UP': -1, 'FL': 1}, {'FS_UP': -1, 'FS_DO': 1}, {'FS_DO': -1, 'FL': 1}, {'FS_DO': -1, 'FS_UP': 1}],
        'rates': {'ks_V': 10 / (72 * pow(60, 2)), 'kd_V': 1 / (72 * pow(60, 2)), 'ke_F_LS_UP': ke_F_LS_UP, 'ke_F_LS_DO': ke_F_LS_DO, 'ke_F_SL_UP': ke_F_SL_UP, 'ke_FS_UP_DO': ke_FS_UP_DO, 'ke_F_SL_DO': ke_F_SL_DO, 'ke_FS_DO_UP': ke_FS_DO_UP},
        'initial_state': {'V': 0, 'FL': 0, 'FS_UP': 0, 'FS_DO': 0}
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
    
    _species_degradation = ['NP', 'FM', 'FL', 'FS_UP', 'FS_DO', 'FD', 'EA']
    degradation = {
        'exes': [f'{S} -> 0' for S in _species_degradation],
        'props': [f'{S} * kd_{S}' for S in _species_degradation],
        'deltas': [{S: -1} for S in _species_degradation],
        'rates': {f'kd_{S}': _rates_protein_degradation['EI'] if S == 'EA' else _rates_protein_degradation[S] for S in _species_degradation},
        'initial_state': {S: 0 if S in {'EA'} else 0 for S in _species_degradation}
    }
    
    # Assemble!
    
    flags = ['promoter_binding', 'promoter_binding_pho', 'promoter_unbinding', 'promoter_unbinding_pho', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'exportation', 'jump_diffuse', 'dimerization', 'enzymatic', 'phosphorylation', 'degradation', 'void']
    initial_state = {}
    rates = {}
    for flag in flags:
        exec(f"initial_state.update({flag}['initial_state'])")
        exec(f"rates.update({flag}['rates'])")
    
    stem = BiochemStem(initial_state, rates)
    
    # flags = ['promoter_binding', 'promoter_binding_pho', 'promoter_unbinding', 'promoter_unbinding_pho', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'exportation', 'jump_diffuse', 'dimerization', 'enzymatic', 'phosphorylation', 'degradation', 'void']
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

from Utilities_Twin import instate_state_tor, instate_rate_mat
from Utilities_Twin import make_jump_diffuse_tor_simul
from Utilities_Twin import interpolator
from Utilities_Twin import plotful
from Cell_Space import cell_placement, cell_distance, cell_slay, cell_neighborhood
from Cell_Space import make_initial_pattern, make_rho_mat, make_sigma_vet
from Utilities_Twin import make_paras, make_para_fun, retrieve_paras
from Utilities_Twin import make_simulator_ready

para_set_raw = {
    'N_N': (100, (0, 1000)), 'G_G': (200, (0, 1000)), 'FC_N': (500, (0, 1000)), 'G_EA': (750, (0, 1000)),
    'G_N': (50, (0, 1000)), 'N_G': (400, (0, 1000)), 'FC_G': (500, (0, 1000)), 'N_EA': (750, (0, 1000)),
    'N_MRNA': (50, (0, 250)), 'G_MRNA': (50, (0, 250)), 'N': (200, (0, 1000)), 'G': (200, (0, 1000)),
    'td_FC': (7200, (300, 28800)), 'td_F_MS': (7200, (300, 28800)), 'td_FL': (7200, (300, 28800)),
    'tau_C': (450, (30, 5*900)), 'tau_M': (450, (30, 2*2100)), 'tau_L': (450, (30, 5*900)), 'tau_S': (450, (30, 2*2100)),
    'tau_ef_EA': (17100, (10*30, 43200)), 'tau_eb_EA': (300, (30, 43200)), 'tau_pf_NP': (1710, (10*30, 43200)), 'tau_pb_NP': (171, (30, 43200)),
    'chi_auto': (0.5, (0, 1))
}
para_set_mode = 0 # {0, 1} # {'No Remap', 'Remap: [A, B] ---->>>> [0, 1]'}
para_set, para_set_true = make_paras(para_set_raw = para_set_raw, para_set_mode = para_set_mode, verbose = True)

#%%# Simulation [Function]

def simulator(parameter_set, parameter_set_true, parameter_set_mode, species = ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP'], steps = 10000, cells = 3, cellulate = (2, 1), time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.2, cell_layers = 2, layer_cells = 1, faces = 2*3, seed = None, jump_diffuse_seed = 0, sup_comp = False, verbose = False):
    
    assert cells == cell_layers * layer_cells + 1, f"We have an ill-defined number of 'cells'!\n\t{cells} ~ {cell_layers * layer_cells + 1}"
    
    if sup_comp: verbose = False # Enforce!
    
    para_fun = make_para_fun(parameter_set, parameter_set_true, parameter_set_mode)
    stem = construct_stem(parameter_set, parameter_set_true, para_fun, sup_comp, verbose = False)
    pat = {'N_MRNA': (para_fun('N_MRNA'), 0.5), 'G_MRNA': (para_fun('G_MRNA'), 0.5), 'N': (para_fun('N'), 0.5), 'G': (para_fun('G'), 0.5)}
    pat_mode = 'Fish_Bind'
    initial_pattern = make_initial_pattern(pat, pat_mode, verbose, species = list(stem.assembly['species'].values()), cells = cells, seed = seed)
    
    cell_location_mat, cell_size_mat = cell_placement(cell_layers, layer_cells, verbose)
    cell_hood_dit, dot_mat = cell_distance(cell_location_mat, verbose)
    cell_hood_dit, dot_mat = cell_slay(cellulate, cell_hood_dit, dot_mat, verbose)
    cell_hoods = cell_neighborhood(cell_hood_dit, dot_mat, cell_layers, layer_cells, verbose)
    comm_classes_portrait = { # Jump-Diffuse Reactions
        0: (['FC_IN -> FM_EX', 'FM_IN -> FM_EX'], cell_hoods[0]),
        1: (['FC_IN -> FL_EX', 'FM_IN -> FL_EX', 'FL_IN -> FM_EX'], cell_hoods[1]),
        2: (['FC_IN -> FS_UP_EX', 'FC_IN -> FS_DO_EX', 'FM_IN -> FS_UP_EX', 'FM_IN -> FS_DO_EX', 'FS_UP_IN -> FM_EX', 'FS_DO_IN -> FM_EX'], cell_hoods[2])
    }
    
    state_tor_PRE = {'N': 0, 'G': 0, 'N_MRNA': 0, 'G_MRNA': 0, 'NP': 0, 'FD': 0, 'EA': 0, 'EI': 1000, 'FC': 0, 'FC_MRNA': 0, 'FM': 0, 'FL': 0, 'FS_UP': 0, 'FS_DO': 0}
    state_tor_EPI = {'N': 0, 'G': 0, 'N_MRNA': 0, 'G_MRNA': 0, 'NP': 0, 'FD': 0, 'EA': 0, 'EI': 1000, 'FC': 0, 'FC_MRNA': 0, 'FM': 0, 'FL': 0, 'FS_UP': 0, 'FS_DO': 0}
    state_tor_V = {'N': 0, 'G': 0, 'N_MRNA': 0, 'G_MRNA': 0, 'NP': 0, 'FD': 0, 'EA': 0, 'EI': 0, 'FC': 0, 'FC_MRNA': 0, 'FM': 0, 'FL': 0, 'FS_UP': 0, 'FS_DO': 0}
    state_tors = {'state_tor_PRE': state_tor_PRE, 'state_tor_EPI': state_tor_EPI, 'state_tor_V': state_tor_V}
    state_tor = instate_state_tor(stem, cell_layers, layer_cells, state_tors, pat_mode, initial_pattern, blank = False, cell_hood_dit = cell_hood_dit)
    
    rates_exclude = ['ks_V', 'kd_V', 'kjd_F_LM', 'ke_F_LS_UP', 'ke_F_LS_DO', 'kjd_F_SM_UP', 'ke_F_SL_UP', 'ke_FS_UP_DO', 'kjd_F_SM_DO', 'ke_F_SL_DO', 'ke_FS_DO_UP', 'kd_FL', 'kd_FS_UP', 'kd_FS_DO'] # Void (Lumen AND Sink_Up AND Sink_Do)
    rho_mat = make_rho_mat(cell_hood_dit, faces, cell_layers, layer_cells, verbose = False)
    sigma_vet = make_sigma_vet(v_0 = 200000, r_1 = 10, verbose = False)
    kd_FC = 1/para_fun('td_FC')
    kd_FM = 1/para_fun('td_F_MS')
    kd_FL = 1/para_fun('td_FL')
    kd_FS_UP = 1/para_fun('td_F_MS')
    kd_FS_DO = 1/para_fun('td_F_MS')
    ksig_C = 1/para_fun('tau_C')
    ksig_M = 1/para_fun('tau_M')
    ksig_L = 1/para_fun('tau_L')
    ksig_S_UP = 1/para_fun('tau_S')
    ksig_S_DO = 1/para_fun('tau_S')
    chi_auto = para_fun('chi_auto')
    rate_mat = instate_rate_mat(stem, cells, parameter_set, parameter_set_true, para_fun, rates_exclude, rho_mat, sigma_vet, blank = False, cell_hood_dit = cell_hood_dit, kd_FC = kd_FC, kd_FM = kd_FM, kd_FL = kd_FL, kd_FS_UP = kd_FS_UP, kd_FS_DO = kd_FS_DO, ksig_C = ksig_C, ksig_M = ksig_M, ksig_L = ksig_L, ksig_S_UP = ksig_S_UP, ksig_S_DO = ksig_S_DO, chi_auto = chi_auto)
    
    instate = {'state_tor': state_tor, 'rate_mat': rate_mat}
    simul = BiochemSimulMuleRapid(stem, instate, steps, cells, seed, verbose = False)
    jump_diffuse_tor, simul = make_jump_diffuse_tor_simul(simul, comm_classes_portrait, jump_diffuse_seed, blank = False)
    epoch_halt_tup = (time_maxi, time_unit) # Stopping Time!
    if sup_comp or verbose: _a = time.time()
    simul.meth_direct(jump_diffuse_tor, epoch_halt_tup)
    if sup_comp or verbose: _b = time.time()
    if sup_comp or verbose: print(f'Simul\nCells = {simul.cells}\tSteps = {simul.steps}\t', _b-_a)
    
    data_inter = interpolator(simul, species, time_mini, time_maxi, time_unit, time_delta, kind = 0, sup_comp = sup_comp, verbose = verbose, err = False, fill_value = 'extrapolate')
    if verbose: plotful(data_inter, species, time_unit)
    trajectory_set = data_inter[1].flatten()
    
    return trajectory_set

#%%# Simulation [Arguments]

parameter_set = para_set
parameter_set_true = para_set_true
parameter_set_mode = para_set_mode
species = ['N', 'G', 'NP'] # ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'FL', 'FS_UP', 'FS_DO', 'EI', 'EA']
steps = 10000000 if sup_comp else 3000000
cells = len(curb) * max(curb) + 1 # 'Late ICM' Plus 'Void (Lumen /\ Sink)'
cellulate = curb # Cell Slay!
time_mini = 0
time_maxi = 48
time_unit = 'Hours'
time_delta = 0.2
cell_layers = len(curb)
layer_cells = max(curb)
faces = 2*3
seed = None # None
jump_diffuse_seed = np.random.default_rng(seed = seed).integers(0, 100)
# sup_comp = False # True
verbose = True

arguments = (parameter_set, parameter_set_true, parameter_set_mode, species, steps, cells, cellulate, time_mini, time_maxi, time_unit, time_delta, cell_layers, layer_cells, faces, seed, jump_diffuse_seed, sup_comp, verbose)

argument_values = arguments[1:len(arguments)]
simulator_ready = make_simulator_ready(simulator, argument_values)

#%%# Simulation [Local Computer Test]

if not sup_comp:
    trajectory_set = simulator(*arguments)

#%%# Inference Procedure [Preparation]

from sbi.inference import prepare_for_sbi
from Utilities_Twin import make_prior
from Utilities_Twin import simul_procedure_loa_comp, simul_procedure_sup_comp, simul_data_load
from Utilities_Twin import restructure, previewer

prior = make_prior(para_set_true, para_set_mode, para_set_sieve = None, verbose = not sup_comp)
simulator_ready, prior = prepare_for_sbi(simulator_ready, prior)

#%%# Simulation /\ Inference Procedure!

tag = f'Act_{act}_Observe_{observe}_{_curb}'
_path = '' if act == 0 else f'Observe_{observe}/'
if sup_comp:
    path = f'/scratch/biochemsim/ramirez/mars_projects/panel/data_bank/{data_path}/' + _path
else:
    path = f'/media/mars-fias/MARS/MARS_Data_Bank/World/Panel/{data_path}/' + _path
post = '_Posterior.pkl'
if act == 0:
    proposal = prior
else:
    _tag = f'Act_{act-1}_Observe_{observe}_{_curb}'
    with open(path + _tag + post, 'rb') as portfolio:
        proposal = pickle.load(portfolio)

#%%# Simulation Procedure: Simul Data Save!

if not inference_prod_activate:
    if sup_comp: # Super Computer
        task = int(sys.argv[1])
        task_simulations = 1000
        safe = True
        collect_zero = simul_procedure_sup_comp(simulator_ready, proposal, task, task_simulations, path, tag, safe)
    else: # Local Computer
        tasks = 2
        task_simulations = 5
        safe = True
        collect_last = simul_procedure_loa_comp(simulator_ready, proposal, tasks, task_simulations, path, tag, safe)

#%%# Inference Procedure: Simul Data Load! [Super Computer <<<<-------->>>> Local Computer]

if inference_prod_activate:
    tasks = None
    theta_set, trajectory_set = simul_data_load(tasks, path, tag, acts)

#%%# Inference Procedure: Simul Data Plot! [Super Computer ---->>>> Local Computer]

if not sup_comp:
    trajectories = [0, 1]
    observations = trajectory_set[trajectories]
    _ = restructure(observations, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, sup_comp = sup_comp, verbose = verbose)

#%%# Inference Procedure: Objective Function!

from ObjectiveFun import ObjectiveFunPanel
from InferenceProd import InferenceProd

train_trials = 10
simul_trials = 10
posts = list()
appraisals = list()
posterior_sample_shape = tuple([250000])
fig_size = (3*7.5, 3*7.5)
verbose = not sup_comp

if inference_prod_activate:
    # Train Loop
    for train_trial in range(train_trials):
        # Preparation
        objective_fun_reference = ObjectiveFunPanel(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, alpha = 1, cellulate = cellulate)
        inference_prod_reference = InferenceProd(objective_fun_reference, theta_set, prior, proposal = None)
        inference_prod_reference.apply(tau_mini = 24, tau_maxi = 48)
        # Synthetic Data
        objective_fun = ObjectiveFunPanel(None, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, alpha = 1, cellulate = cellulate)
        observation = 1*torch.ones(objective_fun_reference.data_objective.shape[2])
        inference_prod_reference.verbose = verbose # View Raw (Simulation) Data!
        inference_prod_reference.synthesizer(observation, posterior_sample_shape, parameter_set_true, simulator_ready, objective_fun, train_trial = train_trial, simul_trials = simul_trials, fig_size = fig_size)
        inference_prod_reference.synth_objective_fun.verbose = verbose # View Synthetic (Simulation) Data!
        appraisal = inference_prod_reference.synth_objective_fun.appraise(train_trial = train_trial)
        appraisals.append(appraisal)
        posts.append(inference_prod_reference.posterior)
        # Restart
        time.sleep(12)
        if train_trial < train_trials - 1:
            del globals()['objective_fun_reference']
            del globals()['inference_prod_reference']
    # Acceptance \/ Rejection
    print(f'Appraisals!\n\t{appraisals}')
    check = len(appraisals) == len(posts)
    mess = 'Oops! Something went wrong!'
    assert check, mess
    where = np.argmax(appraisals)

#%%# Inference Procedure: Posterior Data Save!

if inference_prod_activate:
    posterior = posts[where]
    _tag = f'Act_{act}_Observe_{_observe}_{_curb}'
    if act == 0:
        locus = path + f'Observe_{_observe}/' + _tag + post
    else:
        locus = path + _tag + post
    with open(locus, 'wb') as portfolio:
        pickle.dump(posterior, portfolio)
