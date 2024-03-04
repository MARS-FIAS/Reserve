############################
######## Cell Space ########
############################

#%%# Catalyzer

# import re
import numpy as np
# import numba
# from scipy import interpolate
# import torch
# import sbi
# import time

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Cell Placement

def cell_placement(cell_layers, layer_cells, verbose = False):
    cells = cell_layers * layer_cells # Lumen + Sink := Void
    locations = np.zeros((cells, 3)) # [x, y, z] # Positive Octant!
    edge = 1 # Cell Diameter!
    origin = [0.5 * edge] * 3
    x, y, z = origin
    # Locations!
    for layer in range(cell_layers):
        for cell in range(layer_cells):
            _cell = layer * layer_cells + cell
            location = [x, y, z]
            locations[_cell] = location
            x += edge
            if verbose:
                print(_cell, location)
        x = origin[0]
        z += edge
    # Sizes!
    sizes = np.full((cells, 3), edge)
    placement = (locations, sizes)
    return placement

#%%# Cell Distance

def cell_distance(cell_location_mat, verbose = False):
    
    from scipy import spatial
    
    x = cell_location_mat[:, 0].reshape(-1, 1)
    y = cell_location_mat[:, 1].reshape(-1, 1)
    z = cell_location_mat[:, 2].reshape(-1, 1)
    _x = 1 # {'late': 1, 'mid': 2, 'early': 2}
    _y = 1 # {'late': 1, 'mid': 1, 'early': 2}
    _z = 1 # {'late': 1, 'mid': 1, 'early': 1}
    metric = 'cityblock' # 'euclidean'
    if verbose: print(f"Metric!\t'{metric}'")
    
    _dit_mat_x = spatial.distance.pdist(X = x, metric = metric)
    _dit_mat_y = spatial.distance.pdist(X = y, metric = metric)
    _dit_mat_z = spatial.distance.pdist(X = z, metric = metric)
    
    dit_mat_x = spatial.distance.squareform(_dit_mat_x)
    dit_mat_y = spatial.distance.squareform(_dit_mat_y)
    dit_mat_z = spatial.distance.squareform(_dit_mat_z)
    
    test_x = np.logical_and(np.logical_and(dit_mat_x <= _x, dit_mat_y < _y), dit_mat_z < _z)
    test_y = np.logical_and(np.logical_and(dit_mat_y <= _y, dit_mat_z < _z), dit_mat_x < _x)
    test_z = np.logical_and(np.logical_and(dit_mat_z <= _z, dit_mat_x < _x), dit_mat_y < _y)
    
    dot_mat_x = np.where(test_x, dit_mat_x, 0)
    dot_mat_y = np.where(test_y, dit_mat_y, 0)
    dot_mat_z = np.where(test_z, dit_mat_z, 0)
    
    dot_mat = dot_mat_x + dot_mat_y + dot_mat_z
    # where = np.max(np.argwhere(dot_mat[0] == 1))
    # dot_mat[0:where, -1] = 1
    # dot_mat[-1, 0:where] = 1
    if verbose: print(dot_mat)
    
    cells = cell_location_mat.shape[0]
    cell_hood_dit = {}
    for cell in range(cells):
        hood = np.nonzero(dot_mat[cell])[0]
        cell_hood_dit.update({cell: hood})
        # dot_mat[cell, hood] = 1/hood.shape[0]
    
    distance = (cell_hood_dit, dot_mat)
    
    return distance

#%%# Cell Neighborhood

def _hood_pro_vet(cells, hood):
    for cell in range(cells):
        _hood = hood[cell]
        hood_pro_vet = np.zeros(shape = (cells, ), dtype = np.float)
        if len(_hood) > 0:
            hood_pro_vet[_hood] = 1/_hood.shape[0]
        hood.update({cell: hood_pro_vet.tolist()})
    return hood

def cell_neighborhood(cell_hood_dit, dot_mat, cell_layers, layer_cells, verbose = False):
    import copy
    cell_hood_dit_temp = copy.deepcopy(cell_hood_dit)
    cells = cell_layers * layer_cells # dot_mat.shape[0]
    # Hood 0 # Each 'hood' has its own code piece, careful!
    hood_0 = cell_hood_dit_temp.copy()
    hood_0 = _hood_pro_vet(cells, hood_0)
    # All Hoods!
    cell_hoods = tuple([hood_0])
    return cell_hoods

#%%# Make Rho Mat [ICM Signalling Model | Option Zero]

def make_rho_mat(cell_hood_dit, faces, cell_layers, layer_cells, verbose = False):
    cells = cell_layers * layer_cells
    rho_mat = np.ones((cells, 1))
    for cell in range(cells):
        faces_mem = len(cell_hood_dit[cell])
        rho_mem = faces_mem/faces
        rho = rho_mem
        mess = 'Oops! We made a mistake! Please, check the algorithm.'
        assert 0 <= rho <= 1, mess
        rho_mat[cell] = rho
        if verbose: print(cell_hood_dit[cell], rho)
    return rho_mat

#%%# Make Sigma Vet [Twin Signalling Model | Option Zero]

def make_sigma_vet(v_0 = 200000, r_1 = 10, verbose = False):
    # decrypt = {v_0: (mu)^3, r_1: mu}
    r_0 = np.power(3*v_0/(4*np.pi), 1/3)
    d = 2*np.power(r_0, 2) + 2*r_0*r_1 + np.power(r_1, 2)
    n_0 = 2*np.power(r_0, 2)
    n_1 = 2*r_0*r_1 + np.power(r_1, 2)
    sigma_vet = (n_0/d, n_1/d)
    if verbose:
        print(f'Sigma Vet!\t{sigma_vet}\t{np.sum(sigma_vet)}')
    return sigma_vet

#%%# Blender Draw

def blender_draw(cell_layers, layer_cells, verbose = False):
    
    import sys
    paths = ['/home/mars-fias/anaconda3/lib/python3.8/site-packages', '/home/mars-fias/Documents/Education/Doctorate/Simulator/Resources']
    sys.path.extend(paths)
    from Cell_Space import cell_placement 
    from Cell_Space import coat_placement
    import bpy
    
    bpy.ops.object.select_all(action = 'SELECT')
    bpy.ops.object.delete(use_global = False) # Clean Scene
    
    cell_locations, cell_sizes = cell_placement(cell_layers, layer_cells)
    coat_locations, coat_sizes = coat_placement(cell_locations, cell_sizes)
    
    cells = cell_locations.shape[0]
    edge = np.max(np.diff(a = cell_locations[0:-1], axis = 0))
    
    for cell in range(cells):
        bpy.ops.mesh.primitive_cube_add(size = edge, location = cell_locations[cell])
        bpy.context.object.scale = cell_sizes[cell]
    
    coats = coat_locations.shape[0]
    
    for coat in range(coats):
        bpy.ops.object.empty_add(type = 'CUBE', radius = 0.5 * edge, location = coat_locations[coat])
        bpy.context.object.scale = coat_sizes[coat]

    return None

#%%# Make Initial Pattern

def _draw_initial_patterns_Harsh(coral, cell_layers, layer_cells):
    for key in coral.keys():
        _cot = coral.get(key).reshape((cell_layers, -1))
        zero = np.argwhere(_cot == 0)
        one = np.argwhere(_cot == 1)
        zero_x = zero[:, 1]
        zero_y = zero[:, 0]
        one_x = one[:, 1]
        one_y = one[:, 0]
        plt.scatter(zero_x, zero_y)
        plt.scatter(one_x, one_y)
        plt.xlim(-1, layer_cells)
        plt.ylim(-1, cell_layers)
        plt.axis('equal')
        plt.title(str(coral.get(key)))
        plt.xticks(np.arange(start = 0, stop = layer_cells, dtype = np.int))
        plt.yticks(np.arange(start = 0, stop = cell_layers, dtype = np.int))
        plt.show()
    return None

def _make_initial_pattern_Harsh(pat, cell_layers, layer_cells, verbose):
    cells = cell_layers*layer_cells+1
    mess = 'The current code version does not support neither only one cell layer nor only a single proper cell!'
    assert cell_layers > 1 and cells > 2, mess
    a = np.array([0, 1])
    coral = {key: np.repeat(key, cells-1) for key in a}
    key = max(coral.keys())
    for index in range(key, layer_cells+1):
        if verbose: print(index)
        z = np.repeat(a.reshape((-1, 1)).T, index, 1)
        if verbose: print(z)
        r = (cells - 1) // z.size
        w = np.repeat(z, r, 0).flatten()
        if w.size < cells-1:
            # _w = np.repeat(1-w[-1], cells-1-w.size)
            _w = w[0:cells-1-w.size]
            if verbose: print(_w)
            w = np.concatenate((w, _w))
        assert w.size == cells-1, 'Oops! Something went wrong!'
        if verbose: print(w)
        key += 1
        coral.update({key: w})
        key += 1
        coral.update({key: 1-w})
    if cell_layers > a.size:
        _key = max(coral.keys())
        coral.update({key+_key+1: np.concatenate((np.repeat(key, layer_cells), np.repeat(1-key, cells-1-layer_cells))) for key in a})
    norm = len(coral.keys())
    configuration = int(np.floor(pat*norm))
    if configuration == norm:
        configuration -= 1
        print('Check! Something really weird just happened!')
    if verbose:
        print(configuration)
        _draw_initial_patterns_Harsh(coral, cell_layers, layer_cells)
    initial_pattern = coral.get(configuration) # Stem Initial Pattern!
    return initial_pattern

def _make_initial_pattern_Slide(pat, verbose):
    initial_pattern = pat # Interval [0, 1]
    if verbose: print(f'Initial Pattern Slide Interval!\n\t[Lower, Mid, Upper]\n\t[0, {np.round(pat, 2)}, 1]')
    return initial_pattern

def _make_initial_pattern_IPA(pat, cells, state_tor_NULL, verbose):
    _pat = np.array([index/(cells-1) for index in range(cells-1)])
    initial_pattern = {'aim': np.max(np.argwhere(_pat <= np.array(pat)).flatten()), 'null': state_tor_NULL}
    if verbose: print(f'Initial Pattern IPA!\n\t{initial_pattern}')
    return initial_pattern

def _make_initial_pattern_SAP(pat, cells, seed, verbose):
    if pat is not None:
        mess = "The parameter 'Initial_Pat' must be inactive!"
        raise RuntimeError(mess)
    else: # pat is None
        initial_pattern = np.random.default_rng(seed = seed).choice(a = [0, 1], size = cells-1)
    if verbose: print(f'Initial Pattern SAP!\n\t{initial_pattern}')
    return initial_pattern

def _trial_Fish_Bind(species_lite, lamb, prob, seed, verbose):
    ran = np.random.default_rng(seed = seed)
    fish = ran.poisson(lamb, len(species_lite))
    bind = ran.binomial(fish, prob)
    if verbose: print(species_lite, fish, bind)
    trial = {key: bind[species_lite.index(key)] for key in species_lite}
    return trial

def _make_initial_pattern_Fish_Bind(pat, species, cells, seed, verbose):
    species_lite = list(pat.keys())
    check = all([s in species for s in species_lite])
    mess = 'Oops! All the species must be valid!'
    assert check, mess
    lamb = [value[0] for value in pat.values()]
    prob = [value[1] for value in pat.values()]
    initial_pattern = dict()
    for cell in range(cells):
        trial = _trial_Fish_Bind(species_lite, lamb, prob, seed, verbose)
        initial_pattern.update({cell: trial})
    return initial_pattern

def _trial_Uni_Fish_Bind(species_lite, uni, lamb, prob, seed, verbose):
    ran = np.random.default_rng(seed = seed)
    lamb_lo = [np.maximum(0, np.floor(lamb[_]*(1-uni[_]/100))) for _ in range(len(species_lite))]
    lamb_hi = [np.ceil(lamb[_]*(1+uni[_]/100))+1 for _ in range(len(species_lite))]
    lamb = ran.integers(lamb_lo, lamb_hi, len(species_lite))
    fish = ran.poisson(lamb, len(species_lite))
    bind = ran.binomial(fish, prob)
    if verbose: print(species_lite, lamb, fish, bind)
    trial = {key: bind[species_lite.index(key)] for key in species_lite}
    return trial

def _make_initial_pattern_Uni_Fish_Bind(pat, species, cells, seed, verbose):
    species_lite = list(pat.keys())
    check = all([s in species for s in species_lite])
    mess = 'Oops! All the species must be valid!'
    assert check, mess
    uni = [value[0] for value in pat.values()]
    check = all([isinstance(_, int) and _ >= 0 for _ in uni])
    mess = 'Every value must be a nonnegative integer!'
    assert check, mess
    lamb = [value[1] for value in pat.values()]
    prob = [value[2] for value in pat.values()]
    initial_pattern = dict()
    for cell in range(cells):
        trial = _trial_Uni_Fish_Bind(species_lite, uni, lamb, prob, seed, verbose)
        initial_pattern.update({cell: trial})
    return initial_pattern

def make_initial_pattern(pat, pat_mode, verbose = False, **keywords):
    if pat_mode == 'Harsh':
        cell_layers = keywords['cell_layers']
        layer_cells = keywords['layer_cells']
        initial_pattern = _make_initial_pattern_Harsh(pat, cell_layers, layer_cells, verbose)
    elif pat_mode == 'Slide':
        initial_pattern = _make_initial_pattern_Slide(pat, verbose)
    elif pat_mode == 'IPA':
        cells = keywords['cells']
        state_tor_NULL = keywords['state_tor_NULL']
        initial_pattern = _make_initial_pattern_IPA(pat, cells, state_tor_NULL, verbose)
    elif pat_mode == 'SAP':
        cells = keywords['cells']
        seed = keywords['seed']
        initial_pattern = _make_initial_pattern_SAP(pat, cells, seed, verbose)
    elif pat_mode == 'Fish_Bind':
        species = keywords['species']
        cells = keywords['cells']
        seed = keywords['seed']
        initial_pattern = _make_initial_pattern_Fish_Bind(pat, species, cells, seed, verbose)
    elif pat_mode == 'Uni_Fish_Bind':
        species = keywords['species']
        cells = keywords['cells']
        seed = keywords['seed']
        initial_pattern = _make_initial_pattern_Uni_Fish_Bind(pat, species, cells, seed, verbose)
    else:
        mess = f"The current pattern mode '{pat_mode}' is invalid!"
        raise RuntimeError(mess)
    return initial_pattern

#%%# Section [New]


