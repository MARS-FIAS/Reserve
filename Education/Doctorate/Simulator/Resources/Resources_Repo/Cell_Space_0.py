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
    
    cells = cell_layers * layer_cells + 1 # Lumen + Sink := Void
    locations = np.zeros((cells, 3)) # [x, y, z] # Positive Octant!
    edge = 1 # Cell Diameter!
    origin = [0.5 * edge] * 3
    x, y, z = origin
    
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
    
    sizes = np.full((cells, 3), edge)
    size = np.min(a = locations[0:-1], axis = 0) + np.max(a = locations[0:-1], axis = 0)
    sizes[cells - 1] = size # Void
    location = 0.5 * size
    location[2] *= -1
    locations[cells - 1] = location # Void
    
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
    _cell_hood_dit_temp = copy.deepcopy(cell_hood_dit)
    cells = cell_layers * layer_cells + 1 # dot_mat.shape[0]
    cells_ICM = cells - 1
    cells_EPI = (cell_layers - 1) * layer_cells
    cells_PRE = cells_ICM - cells_EPI
    
    # Hood 0 # Each 'hood' has its own code piece, careful!
    
    cell_hood_dit_temp.update({cells - 1: np.empty(shape = (0, ), dtype = np.int)}) # There is no interaction between cell class and cavity/sink class!
    
    hood_0 = cell_hood_dit_temp.copy()
    hood_0 = _hood_pro_vet(cells, hood_0)
    
    # Hood 1 # Each 'hood' has its own code piece, careful!
    
    cell_hood_dit_temp.update({cells - 1: np.arange(start = 0, stop = cells_PRE, dtype = np.int)}) # There is interaction between cell class and 'cavity'/sink class!
    cell_hood_dit_temp.update({cell: np.array([cells_ICM], dtype = np.int) for cell in range(cells_PRE)})
    cell_hood_dit_temp.update({cell: np.array([], dtype = np.int) for cell in range(cells_PRE, cells_ICM)})
    
    hood_1 = cell_hood_dit_temp.copy()
    hood_1 = _hood_pro_vet(cells, hood_1)
    
    # Hood 2 # Each 'hood' has its own code piece, careful!
    
    cells_sink = [cell for cell in range(cells_ICM) if len(_cell_hood_dit_temp[cell]) < 2*3]
    
    cell_hood_dit_temp.update({cells - 1: np.array(cells_sink, dtype = np.int)}) # There is interaction between cell class and cavity/'sink' class!
    cell_hood_dit_temp.update({cell: np.array([cells_ICM], dtype = np.int) for cell in cells_sink})
    cell_hood_dit_temp.update({cell: np.array([], dtype = np.int) for cell in range(cells_ICM) if cell not in cells_sink})
    
    hood_2 = cell_hood_dit_temp.copy()
    hood_2 = _hood_pro_vet(cells, hood_2)
    
    # All Hoods!
    
    cell_hoods = (hood_0, hood_1, hood_2)
    
    return cell_hoods

#%%# Make Rho Mat [ICM Signalling Model | Option Zero]

def make_rho_mat(cell_hood_dit, faces, cell_layers, layer_cells, verbose = False):
    
    cells = cell_layers * layer_cells + 1
    cells_ICM = cells - 1
    cells_EPI = (cell_layers - 1) * layer_cells
    cells_PRE = cells_ICM - cells_EPI
    
    rho_mat = np.ones((cells, 3)) # (rho_cell, rho_void, rho_sink)
    
    for cell in range(cells_PRE):
        faces_cell = len(cell_hood_dit[cell])
        faces_void = 1
        faces_sink = faces-(faces_cell+faces_void)
        rho_cell, rho_void, rho_sink = faces_cell/faces, faces_void/faces, faces_sink/faces
        rho = (rho_cell, rho_void, rho_sink)
        mess = 'Oops! We made a mistake! Please, check the algorithm.'
        assert np.sum(rho) == 1, mess
        rho_mat[cell] = rho
        if verbose: print(cell_hood_dit[cell], rho)
    
    for cell in range(cells_PRE, cells_ICM):
        faces_cell = len(cell_hood_dit[cell])
        faces_void = 0
        faces_sink = faces-(faces_cell+faces_void)
        rho_cell, rho_void, rho_sink = faces_cell/faces, faces_void/faces, faces_sink/faces
        rho = (rho_cell, rho_void, rho_sink)
        mess = 'Oops! We made a mistake! Please, check the algorithm.'
        assert np.sum(rho) == 1, mess
        rho_mat[cell] = rho
        if verbose: print(cell_hood_dit[cell], rho)
    
    return rho_mat

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

#%%# Section [New]


