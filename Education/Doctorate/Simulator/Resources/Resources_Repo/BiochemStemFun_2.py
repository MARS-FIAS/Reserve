#%%# Catalyzer

# Libraries

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue

import numba
import re

import pandas as pd
from scipy import stats
from scipy import optimize # Curve Fitting
from scipy import interpolate
import math
import pickle
import sys
import gc
import time # Unnecessary!

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Biochemical System Construction

# Regulation [Components AND Interactions]

regulation_transcription = { # ([0|1] = [Up|Down]-Regulate, Positive Integer = Transcription Cooperativity)
    'N': {'N': (0, 4), 'G': (1, 4), 'FI': (0, 2)},
    'G': {'N': (1, 4), 'G': (0, 4), 'FI': (1, 2)},
    'EA': {'N': (1, 3), 'G': (0, 3)}
}

# Species [Specs] # Transmembrane

_species_promoter_state = ['I', 'A'] # {'I': 'Inactive', 'A': 'Active'}
_species_transcription = ['N', 'G', 'FI']
_species_translation = _species_transcription.copy()
_species_translation.extend(['FR', 'EI'])

_species = {
    'promoter_state': [S + '_' + _ for S in _species_translation for _ in _species_promoter_state], # Promoter State Dynamics
    'transcription': [S + '_MRNA' for S in _species_transcription], # Explicit Transcription
    'translation': _species_translation, # Explicit Translation
    'jump_diffuse': ['FE'],
    'receptor': ['FR'],
    'ligand_bound': ['L'],
    'enzymatic': ['EA']
}

# species

# Rate Constants

# None

diffusion_coefficients = { # pow(microns, 2)/seconds # pow(length, 2)/time # Protein diffusion constant
    'N': 10e-12,
    'G': 10e-12,
    'FI': 10e-12,
    'FE': 10e-12,
    'EA': 10e-12
}

protein_cross_sections = { # Nanometers
    'N': 10e-9,
    'G': 10e-9,
    'FI': 10e-9,
    'FE': 10e-9,
    'EA': 10e-9
}

binding_sites = list(regulation_transcription.keys())

_cell_radius = 1 # # Nanometers
cell_radius = 1000e-9*_cell_radius
cell_volume = 4*math.pi*pow(cell_radius, 3)/3 # pow(meters, 3)

half_activation_threshold = 250
half_repression_threshold = 1000
_rates_promoter_binding = {S: 4*math.pi*protein_cross_sections[S]*diffusion_coefficients[S]/cell_volume for S in binding_sites}
_rates_promoter_unbinding = {P+'_'+S: 10*half_activation_threshold*_rates_promoter_binding[P] if S == 'A' else 10*half_repression_threshold*_rates_promoter_binding[P] for P in binding_sites for S in _species_promoter_state}

_MRNA_lifetime = 4 # {1, ..., 8} # Hours
MRNA_lifetime = _MRNA_lifetime*pow(60, 2) # Seconds
MRNA_copy_number = 250
synthesis_spontaneous = 0.2 # [0, 1] # 1 - synthesis
synthesis = 1 - synthesis_spontaneous
_rates_MRNA_synthesis_spontaneous = synthesis_spontaneous*MRNA_copy_number/MRNA_lifetime
_rates_MRNA_synthesis = synthesis*MRNA_copy_number/MRNA_lifetime
_rates_MRNA_degradation = 1/MRNA_lifetime

_protein_lifetime = 2 # {1, ..., 8} # Hours
protein_lifetime = _protein_lifetime*pow(60, 2) # Seconds
protein_copy_numbers = { # 1000
    'N': 4, # Proteins Per MRNA
    'G': 4, # Proteins Per MRNA
    'FI': 1000,
    'FR': 1000,
    'EI': 1000
}
_rates_protein_synthesis = {S: protein_copy_numbers[S]/protein_lifetime for S in list(protein_copy_numbers.keys())}
_rates_protein_degradation = 1/protein_lifetime

# Reactions [template = {'exes': [], 'props': [], 'deltas': [{}], 'rates': {}, 'initial_state': {}}]

promoter_binding = { # {P = Promoter}_{B = Binding Site}_{C = Cooperativity}
    'exes': [f'{B} + {P}_{B}_{C} -> {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])],
    'props': [f'{B} * {P}_{B}_{C} * kb_{P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])],
    'deltas': [{B: -1, f'{P}_{B}_{C}': -1, f'{P}_{B}_{C+1}': 1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': 1 if C+1 == regulation_transcription[B][P][1] else 0} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])],
    'rates': {f'kb_{P}_{B}_{C+1}': _rates_promoter_binding[B] for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])},
    'initial_state': {f'{P}_{B}_{C}': 0 if C != 0 else 1 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]+1)}
}

pub = 5 # Promoter Unbinding [Coefficient]
promoter_unbinding = {
    'exes': [f'{P}_{B}_{C+1} -> {P}_{B}_{C} + {B}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)],
    'props': [f'ku_{P}_{B}_{C} * {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)],
    'deltas': [{B: 1, f'{P}_{B}_{C}': 1, f'{P}_{B}_{C+1}': -1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': -1 if C+1 == regulation_transcription[B][P][1] else 0} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)],
    'rates': {f'ku_{P}_{B}_{C}': _rates_promoter_unbinding[B+'_A']/pow(pub, C) if regulation_transcription[B][P][0] == 0 else _rates_promoter_unbinding[B+'_I']/pow(pub, C) for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)},
    'initial_state': {f'{P}_{B}_{C}': 0 if C != 0 else 1 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1], -1, -1)}
}

MRNA_synthesis_spontaneous = {
    'exes': [f'0 -> {S}' for S in _species['transcription']],
    'props': [f'(1 - {S}_I) * kss_{S}_MRNA' for S in _species_transcription],
    'deltas': [{S: 1} for S in _species['transcription']],
    'rates': {f'kss_{S}': _rates_MRNA_synthesis_spontaneous for S in _species['transcription']},
    'initial_state': {S: 0 for S in _species['transcription']}
}

MRNA_synthesis = {
    'exes': [f'{S}_A -> {S}_MRNA' for S in _species_transcription],
    'props': [f'{S}_A * (1 - {S}_I) * ks_{S}_MRNA' for S in _species_transcription],
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
    'initial_state': {f'{S}_{_}': 0 for S in _species['translation'] for _ in _species_promoter_state}
}

protein_degradation = {
    'exes': [f'{S} -> 0' for S in _species['translation']],
    'props': [f'{S} * kd_{S}' for S in _species['translation']],
    'deltas': [{S: -1} for S in _species['translation']],
    'rates': {f'kd_{S}': _rates_protein_degradation for S in _species['translation']},
    'initial_state': {S: 0 for S in _species['translation']}
}

# None

jump_diffuse = { # Two-Step Process?
    'exes': ['FI -> FE'],
    'props': ['FI * kjd_FE'],
    'deltas': [{'FI': -1, 'FE': 1}],
    'jump_diffuse_deltas': [{'FI': 1, 'FE': 1}],
    'rates': {'kjd_FE': _rates_protein_synthesis['FI']},
    'initial_state': {'FI': 0, 'FE': 0}
}

ligand_bound = { # ligand_unbound
    'exes': ['FE + FR -> L', 'L -> FR + FE'],
    'props': ['FE * FR * klb_L', 'L * klu_L'],
    'deltas': [{'FE': -1, 'FR': -1, 'L': 1}, {'FE': 1, 'FR': 1, 'L': -1}],
    'rates': {'klb_L': _rates_promoter_binding['EA'], 'klu_L': _rates_promoter_binding['EA']/10},
    'initial_state': {'FE': 0, 'FR': 0, 'L': 0}
}

enzymatic = { # Reverse # [Forward | Backward]
    'exes': ['L + EI -> EA', 'EA -> EI'],
    'props': ['L * EI * kef_EA', 'EA * keb_EA'],
    'deltas': [{'L': -1, 'EI': -1, 'EA': 1}, {'L': 1, 'EI': 1, 'EA': -1}],
    'rates': {'kef_EA': 100*_rates_promoter_binding['EA'], 'keb_EA': _rates_promoter_binding['EA']},
    'initial_state': {'L': 0, 'EI': 0, 'EA': 0}
}

_species_degradation = ['FE', 'L', 'EA']
degradation = {
    'exes': [f'{S} -> 0' for S in _species_degradation],
    'props': [f'{S} * kd_{S}' for S in _species_degradation],
    'deltas': [{S: -1} for S in _species_degradation],
    'rates': {f'kd_{S}': _rates_protein_degradation/10 for S in _species_degradation},
    'initial_state': {S: 0 for S in _species_degradation}
}

# None

flags = ['promoter_binding', 'promoter_unbinding', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'jump_diffuse', 'ligand_bound', 'enzymatic', 'degradation']
initial_state = {}
rates = {}
for flag in flags:
    exec(f"initial_state.update({flag}['initial_state'])")
    exec(f"rates.update({flag}['rates'])")

stem = BiochemStem(initial_state, rates)

flags = ['promoter_binding', 'promoter_unbinding', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'jump_diffuse', 'ligand_bound', 'enzymatic', 'degradation']
for flag in flags:
    print(flag)
    indices = eval(f"range(len({flag}['exes']))")
    for index in indices:
        print(index)
        name = eval(f"{flag}['exes'][{index}]")
        prop_fun = eval(f"{flag}['props'][{index}]")
        delta = eval(f"{flag}['deltas'][{index}]")
        if flag == 'jump_diffuse':
            jump_diffuse_delta = eval(f"{flag}['jump_diffuse_deltas'][{index}]")
        else:
            jump_diffuse_delta = None
        stem.add_reaction(name, prop_fun, delta, verbose = False, jump_diffuse = jump_diffuse_delta)

stem.assemble()

#%%# Biochemical System Simulation

# stem
steps = 100000
cells = np.power(2, 1)
seed = 52

jump_diffuse_mat = np.array([1, 0])

rapid = BiochemSimulMuleRapid(stem, steps, cells, seed, False)

#%%# Rapid

_a = time.time()
rapid.meth_direct(jump_diffuse_mat)
_b = time.time()
print(f'Rapid\nCells = {rapid.cells}\tSteps = {rapid.steps}\t', _b-_a)

#%%# Plot

for cell in range(cells): # trajectory # range(trajectories)

    s = [51]
    x = rapid.epoch_mat[:rapid.step_cells[cell], cell]
    y = rapid.state_tor[:rapid.step_cells[cell], s, cell]
    plt.title(f'Cell\n{cell}')
    plt.xlim(0, np.nanmax(rapid.epoch_mat))
    plt.ylim(-0.25, np.nanmax(rapid.state_tor[:rapid.step_cells[cell], s])+0.25)
    plt.plot(x, y, drawstyle = 'steps-post', linestyle = '--')
    plt.show()
