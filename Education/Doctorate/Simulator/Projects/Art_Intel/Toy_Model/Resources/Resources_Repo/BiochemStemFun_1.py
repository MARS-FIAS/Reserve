######## Name
######## BiochemStemFun

######## Requirements
######## {Modules}

#%%# Biochemical System Construction

# Regulation [Components AND Interactions]

regulation_transcription = { # ([0|1] = [Up|Down]-Regulate, Positive Integer = Transcription Cooperativity)
    'N': {'N': (0, 4), 'G': (1, 4), 'FI': (0, 2), 'FRM': (1, 2)},
    'G': {'N': (1, 4), 'G': (0, 4), 'FI': (1, 2), 'FRM': (0, 2)},
    'EA': {'N': (1, 3), 'G': (0, 3)}
}

# Species [Specs] # Transmembrane

_species_promoter_state = ['I', 'A'] # {'I': 'Inactive', 'A': 'Active'}
_species_transcription = ['N', 'G', 'EI', 'FI', 'FRM']
_species_translation = _species_transcription

_species = {
    'promoter_state': [S + '_' + _ for S in _species_translation for _ in _species_promoter_state], # Promoter State Dynamics
    'transcription': [S + '_MRNA' for S in _species_transcription], # Explicit Transcription
    'translation': _species_translation, # Explicit Translation
    'jump_diffuse': ['FE'],
    'ligand_bound': ['FRL'],
    'dimerize': ['FRD'],
    'enzymatic': ['EA']
}

# species

# Rate Constants

# None

# Reactions [template = {'exes': None, 'props': None, 'deltas': None, 'rates': None, 'initial_states': None}]

binding_sites = list(regulation_transcription.keys())

promoter_binding = { # {P = Promoter}_{B = Binding Site}_{C = Cooperativity}
    'exes': [f'{B} + {P}_{B}_{C} -> {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])]
}

promoter_unbinding = {
    'exes': [f'{P}_{B}_{C+1} -> {P}_{B}_{C} + {B}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)]
}

MRNA_synthesis_spontaneous = {
    'exes': [f'0 -> {S}' for S in _species['transcription']]
}

MRNA_synthesis = {
    'exes': [f'{S}_A -> {S}_MRNA' for S in _species_transcription]
}

MRNA_degradation = {
    'exes': [f'{S} -> 0' for S in _species['transcription']]
}

protein_synthesis = {
    'exes': [f'{S}_MRNA -> {S}' if S in _species_transcription else f'{S}_A -> {S}' for S in _species['translation']]
}

protein_degradation = {
    'exes': [f'{S} -> 0' for S in _species['translation']]
}

# None

jump_diffuse = {
    'exes': ['FI -> FE']
}

ligand_bound = { # ligand_unbound
    'exes': ['FE + FRM -> FRL', 'FRL -> FRM + FE']
}

dimerize_spontaneous = { # Reverse (Dissociate?)
    'exes': ['FRM + FRM -> FRD', 'FRD -> FRM + FRM']
}

dimerize = { # Reverse (Dissociate?)
    'exes': ['FRL + FRL -> FRD', 'FRD -> FRL + FRL']
}

enzymatic = { # Reverse
    'exes': ['FRD + EI -> EA', 'EA -> EI']
}

degradation = {
    'exes': [f'{S} -> 0' for S in ['FE', 'FRL', 'FRD', 'EA']]
}
