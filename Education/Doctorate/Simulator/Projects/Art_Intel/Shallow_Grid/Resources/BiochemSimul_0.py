######## Name
######## BiochemSimulMuleRapid

######## Requires
######## {Modules}

import numpy as np
import numba

class BiochemSimulMuleRapid:
    
    """Class 'BiochemSimul' Illustration!
    It is an essential part of the 'BiochemAnalysis' class!
    
    ########
    Attributes
        
    
    ########
    Methods
        
    
    """
    
    def __init__(self, stem, instate, steps, cells, species_objective = None, seed = None, verbose = True):
        """
        
        
        """
        if not stem._assembled:
            raise RuntimeError('The biochemical system is not assembled! Check it again!') # Stop!
        elif not all([key in list(instate.keys()) for key in ['state_tor', 'rate_mat']]):
            raise RuntimeError('A valid instantiation must have 2 keys!') # Stop!
        self.stem = stem
        self.instate = instate
        self.steps = steps
        self.cells = cells
        self.species_objective = species_objective
        self.seed = seed
        self.verbose = verbose
    
    def __repr__(self):
        portrait = f'<{self.__class__.__name__}({self.stem.__class__.__name__}, {self.steps}, {self.cells})>'
        return portrait
    
    def __str__(self):
        portrait = repr(self)
        return portrait
    
    def _ran_tor(self):
        self.ran_tor = np.random.default_rng(seed = self.seed).uniform(size = (self.steps, 2, self.cells))
        self.ran_tor[:, 0] = -np.log(self.ran_tor[:, 0])
        return self
    
    def _epoch_mat(self):
        self.epoch_mat = np.full(shape = (self.steps, self.cells), fill_value = np.nan)
        self.epoch_mat[0] = 0
        return self
    
    def _state_tor(self):
        if self.species_objective is None:
            species = list(self.stem.assembly['species'].values())
            self.species_objective = species
        else:
            check = [s in list(self.stem.assembly['species'].values()) for s in self.species_objective]
            mess = f"The list 'stem.assembly['species']' is not consistent with the 'species_objective' list!\n\t{np.array(self.species_objective)[np.invert(check)]}"
            assert all(check), mess
            species = self.species_objective
        self.species_objective_indices = [list(self.stem.assembly['species'].values()).index(s) for s in species]
        self.state_tor = np.zeros(shape = (1, len(self.stem.assembly['species']), self.cells), dtype = np.uint16)
        self.state_tor_objective = np.zeros(shape = (self.steps, len(species), self.cells), dtype = np.uint16)
        if self.instate['state_tor'] is None:
            initial_state = np.array(list(self.stem.initial_state.values())).reshape(-1, 1)
            self.state_tor[0, ...] = np.repeat(initial_state, self.cells, 1)
            self.state_tor_objective[0, ...] = np.repeat(initial_state[self.species_objective_indices, 0], self.cells, 1)
        else:
            if self.state_tor[0, ...].shape != self.instate['state_tor'].shape:
                raise RuntimeError('Wrong! The current instantiation is not compatible with our framework!') # Stop!
            self.state_tor[0, ...] = self.instate['state_tor']
            self.state_tor_objective[0, ...] = self.instate['state_tor'][self.species_objective_indices, :]
        return self
    
    def _rate_mat(self):
        self.rate_mat = np.zeros(shape = (len(self.stem.rates.values()), self.cells))
        if self.instate['rate_mat'] is None:
            rates = np.array(list(self.stem.rates.values())).reshape(-1, 1)
            self.rate_mat = np.repeat(rates, self.cells, 1)
        else:
            if self.rate_mat.shape != self.instate['rate_mat'].shape:
                raise RuntimeError('Wrong! The current instantiation is not compatible with our framework!') # Stop!
            self.rate_mat = self.instate['rate_mat']
        return self
    
    def _pre(self):
        pre = self.state_tor[0, ...]
        return pre

    def _alps(self):
        if self.verbose:
            fun = '__alps__'
            arcs = ['cell', 'rate_mat', 'species', 'alp_mat', 'indices']
            print(fun, '\n\t', arcs)
        if '_alps' not in globals():
            global __alps__
            @numba.njit
            def __alps__(cell, rate_mat, species, alp_mat, indices):
                for _cell in range(species.shape[1]):
                    if species.shape[1] > 1:
                        cell = _cell
                    _rate_mat = rate_mat[:, cell]
                    _species = species[:, _cell]
                    for index in indices:
                        alp_mat[index, _cell] = prop_funs(index, _rate_mat, _species)
                alp_mat[0] = np.sum(alp_mat, axis = 0)
                for j in range(alp_mat.shape[1]):
                    alp_mat[1:, j] = np.cumsum(alp_mat[1:, j])
                alp_mat[1:] = alp_mat[1:] / alp_mat[0]
                return alp_mat
        return __alps__
    
    def _refresh_tau(self):
        if self.verbose:
            fun = '__refresh_tau__'
            arcs = ['ran_tor', 'step', 'cell', 'alps']
            print(fun, '\n\t', arcs)
        if '_refresh_tau' not in globals():
            global __refresh_tau__
            @numba.njit
            def __refresh_tau__(ran_tor, step, cell, alps):
                tau = ran_tor[step, 0, cell] / alps[0]
                return tau
        return __refresh_tau__

    def _refresh_delta(self):
        if self.verbose:
            fun = '__refresh_delta__'
            arcs = []
            print(fun, '\n\t', arcs)
        if '_refresh_delta' not in globals():
            global __refresh_delta__
            @numba.njit
            def __refresh_delta__(ran_tor, step, cell, alps, delta_mat):
                clues_boa = ran_tor[step, 1, cell] <= alps[1:]
                # _clues = np.indices(clues_boa.shape, dtype = np.int32)
                # clues = _clues[0]
                clues = np.arange(clues_boa.size, dtype = np.int32)
                temp = np.where(clues_boa, clues, 10*clues.shape[0])
                inventory = np.nanmin(temp) # list(np.nanmin(temp, 0))
                delta_trap = delta_mat[1:]
                delta = np.transpose(delta_trap[inventory])
                return delta, inventory
        return __refresh_delta__
    
    def _refresh_rate_mat(self):
        if self.verbose:
            fun = '__refresh_rate_mat__'
            arcs = []
            print(fun, '\n\t', arcs)
            if '_refresh_rate_mat' not in globals():
                global __refresh_rate_mat__
                @numba.njit
                def __refresh_rate_mat__():
                    return None
        return __refresh_rate_mat__
    
    def _streamer(self, role):
        if role == 'make':
            if '_streamer_make' not in globals():
                global _make
                @numba.njit
                def _make(cells, prop_funs):
                    shape = (cells, 2)
                    stream_epoch = np.full(shape, np.nan)
                    shape = (cells, prop_funs)
                    stream_alps = np.full(shape, np.nan)
                    ret = (stream_epoch, stream_alps)
                    return ret
            return _make
        if role == 'put':
            if '_streamer_put' not in globals():
                global _put
                @numba.njit
                def _put(stream_epoch, stream_alps, epoch, cell, alps):
                    stream_epoch[cell] = [epoch, cell] # np.array([epoch, cell])
                    stream_alps[cell] = alps
                    ret = (stream_epoch, stream_alps)
                    return ret
            return _put
        if role == 'get':
            if '_streamer_get' not in globals():
                global _get
                @numba.njit
                def _get(stream_epoch, stream_alps):
                    here = np.argmin(stream_epoch[:, 0])
                    mine_epoch = stream_epoch[here]
                    mine_alps = stream_alps[here]
                    epoch = mine_epoch[0]
                    cell = np.int(mine_epoch[1])
                    alps = mine_alps.copy()
                    stream_epoch[here] = np.array([np.inf]*2)
                    stream_alps[here] = np.array([np.inf]*mine_alps.shape[0])
                    ret = (stream_epoch, stream_alps, epoch, cell, alps)
                    return ret
            return _get
        else:
            return None
    
    def jump_diffuse_assemble(self, comm_classes_portrait, jump_diffuse_seed):
        # Zero
        comm_classes = list(comm_classes_portrait.keys())
        comm_class_reactions = [value[0] for value in comm_classes_portrait.values()]
        comm_class_neighborhoods = [value[1] for value in comm_classes_portrait.values()]
        #
        reactions = list(self.stem.assembly['reactions'].values())[1:]
        jump_diffuse_vet = self.stem.assembly['jump_diffuse_vet']
        jump_diffuse_reactions = [reaction for reaction in reactions if jump_diffuse_vet[reactions.index(reaction)]]
        #
        _check = np.sort(comm_classes)
        check = np.arange(0, np.max(_check)+1)
        assert np.logical_and(np.array_equal(check, _check), np.max(_check) < np.power(2, 8)-1), f"Error! Comm classes must be defined as consecutive integers starting with 0 (and ending with {np.power(2, 8)-2}), always!\nComm Class ~ [0, 1, 2, 3, 4, ..., {np.power(2, 8)-2}]\nThe comm class '{np.power(2, 8)-1}' is reserved for non-jump-diffuse reactions!"
        #
        _check = []
        for comm_class in comm_classes:
            _check.extend(comm_class_reactions[comm_class])
        check = [reaction in _check for reaction in jump_diffuse_reactions]
        assert all(check), f'Oops! We must specify a comm class for every jump-diffuse reaction within our current stem! We are missing several reactions!\n\t{np.array(jump_diffuse_reactions)[np.invert(check)]}'
        #
        for comm_class in comm_classes:
            assert type(comm_class) == int, "Error! Comm classes must be integers!"
            for reaction in comm_class_reactions[comm_class]:
                assert reaction in jump_diffuse_reactions, f"Error! The reaction '{reaction}' given by the comm class '{comm_class}' is not yet defined for the current jump-diffuse stem!"
        #
        _jump_diffuse_comm_classes = [comm_class for reaction in jump_diffuse_reactions for comm_class in comm_classes if reaction in comm_class_reactions[comm_class]]
        jump_diffuse_comm_classes = np.full(jump_diffuse_vet.shape, np.power(2, 8)-1, dtype = np.uint8) # [0, ..., 255]
        jump_diffuse_comm_classes[jump_diffuse_vet] = _jump_diffuse_comm_classes
        # Zero
        # One
        comm_class_tor = np.full(shape = (len(comm_classes), self.cells, self.cells), fill_value = np.nan)
        #
        for comm_class in comm_classes:
            temp = np.array(object = list(comm_class_neighborhoods[comm_class].values()), dtype = np.float64, ndmin = 2)
            assert temp.shape == (self.cells, self.cells), f"The neighborhood '{comm_class}' has an invalid shape!\nThe only valid shape is '{(self.cells, self.cells)}'!"
            comm_class_tor[comm_class] = temp
        #
        check = np.argwhere(np.logical_and(np.isclose(comm_class_tor.sum(2), 0), np.isclose(comm_class_tor.sum(2), 1)))
        _mess = [
            f"The probability vector for the following '[neigborhood, cell]' is wrong! Sum of all entries must equal exactly 0 or 1, but nothing between these values!\n{check}\n\nDescription 'comm_class_tor'\n\t",
            "comm_class_tor[i]\n\t\tRespective neighborhood (integer) identity!\n\t",
            "comm_class_tor[i][j]\n\t\tProbability vector!\n\t\tJump from cell 'j' to each cell within our current neighborhood!\n\t",
            "comm_class_tor[i][j][k]\n\t\tProbability of jumping from cell 'j' to cell 'k'!"
        ]
        mess = ''.join(_mess)
        # print(mess)
        print('i := comm_class\nj := cell\nk := neighbor')
        assert np.all(np.logical_or(np.isclose(comm_class_tor.sum(2), 0), np.isclose(comm_class_tor.sum(2), 1))), mess
        # One
        # Two
        descript = ['(Comm Class, Cell)', 'Neighbors', 'Jump-Diffuse Probabilities']
        comm_class_summary = {'Descript': descript}
        decor = '########'
        #
        print(f'{decor}\n\nComm Class\n\n\t{descript[0]}\n{descript[1]}\n{descript[2]}\n\n{decor}\n')
        for comm_class in range(comm_class_tor.shape[0]):
            print(f'{decor}\n\nComm Class\t{comm_class}\n')
            for cell in range(comm_class_tor.shape[1]):
                neighbors = np.argwhere(comm_class_tor[comm_class, cell] != 0).ravel().tolist()
                jump_diffuse_probabilities = comm_class_tor[comm_class, cell, neighbors].tolist()
                print(f'\t{(comm_class, cell)}\n{np.vstack((neighbors, jump_diffuse_probabilities))}\n')
                # print(f'\t{cell}\n\t\t{neighbors}\n\t\t{jump_diffuse_probabilities}')
                comm_class_summary.update({(comm_class, cell): (neighbors, jump_diffuse_probabilities)})
            print(f'{decor}\n')
        # Two
        # Last
        jump_diffuse_tor = np.full(shape = (comm_class_tor.shape[0], self.steps*self.cells, self.cells), fill_value = np.power(2, 8)-1, dtype = 'uint8')
        _decor = '@@@@@@@@'
        a = np.arange(self.cells)
        print(f'{decor}\n\nCells {a}')
        for comm_class in range(comm_class_tor.shape[0]):
            for cell in range(comm_class_tor.shape[1]):
                print(f'\n(Comm Class, Cell) {(comm_class, cell)}')
                p = comm_class_tor[comm_class, cell]
                print(f'p = {p}')
                try:
                    seed = jump_diffuse_seed + self.cells + cell
                    _jump_diffuse_tor = np.random.default_rng(seed = seed).choice(a = a, size = self.steps*self.cells, replace = True, p = p)
                except ValueError:
                    mess = f'{_decor} No neighbors! {_decor}'
                    print(mess)
                    continue
                print(_jump_diffuse_tor)
                print(np.unique(_jump_diffuse_tor).tolist())
                print(comm_class_summary[(comm_class, cell)][0])
                assert np.unique(_jump_diffuse_tor).tolist() == comm_class_summary[(comm_class, cell)][0], 'Oops! Something went wrong!'
                jump_diffuse_tor[comm_class, :, cell] = _jump_diffuse_tor
        print(f'\n{decor}\n')
        for comm_class in range(comm_class_tor.shape[0]):
            print(f'Comm Class {comm_class}\n\n{jump_diffuse_tor[comm_class]}\n')
        # Last
        self.jump_diffuse_comm_classes = jump_diffuse_comm_classes
        self.comm_class_tor = comm_class_tor
        self.comm_class_summary = comm_class_summary
        return jump_diffuse_tor
    
    def _meth_direct(self):
        if self.verbose:
            fun = '__meth_direct__'
            arcs = []
            print(fun, '\n\t', arcs)
        if '_meth_direct' not in globals():
            global __meth_direct__
            @numba.njit
            def __meth_direct__(steps, cells, step_cells, stream_epoch, stream_alps, epoch_mat, pre, state_tor, state_tor_objective, species_objective_indices, ran_tor, rate_mat, indices, delta_mat, jump_diffuse_vet, jump_diffuse_comm_classes, jump_diffuse_tor, jump_diffuse_delta_mat, epoch_halt):
                halt_flag = False # Stopping Time Flag!
                for step in range(1, steps*cells):
                    # Alpha [Start]
                    stream_epoch, stream_alps, epoch, cell, alps = _streamer_get(stream_epoch, stream_alps)
                    step_cell = step_cells[cell]
                    epoch_mat[step_cell, cell] = epoch
                    delta, inventory = _refresh_delta(ran_tor, step_cell, cell, alps, delta_mat)
                    ## Epoch Halt [Start]
                    if epoch > epoch_halt:
                        halt_flag = True
                    ## Epoch Halt [Final]
                    #
                    # print(f'Step {step}\n\tCell {cell}\t{list(self.stem.reactions.keys())[inventory]}')
                    # if (cell == 0 and inventory == 0) or (cell == 1 and inventory == 1):
                    #     print(f'\tAlps {cell}\n\t{alps}\n\tAlps {(cell+1)%2}\n\t{stream_alps[(cell+1)%2]}')
                    #
                    ## Jump Diffuse [Start]
                    if jump_diffuse_vet[inventory] and jump_diffuse_tor is not None:
                        ####
                        # if jump_diffuse_tor.size == cells:
                        #     jump_diffuse_where = np.copy(jump_diffuse_tor)
                        # else:
                        #     # jump_diffuse_where = np.array(jump_diffuse_tor[step])
                        #     pass
                        jump_diffuse_where = np.copy(jump_diffuse_tor[jump_diffuse_comm_classes[inventory], step])
                        ####
                        where = jump_diffuse_where[cell]
                        epoch_mat[step_cells[where], where] = epoch
                        temp = pre[:, where] + jump_diffuse_delta_mat[inventory+1]
                        state_tor[0, :, where] = temp
                        state_tor_objective[step_cells[where], :, where] = temp[species_objective_indices]
                        ####
                        step_cells[where] += 1
                        if np.any(step_cells >= steps):
                            # self.step_cells = step_cells
                            break
                        pre[:, where] = temp
                        ####
                        ####
                        stream_epoch[where] = np.inf
                        stream_alps[where] = np.inf
                        ####
                        ####
                        _species = pre[:, where].copy()
                        species = _species.reshape((-1, 1))
                        alp_mat = np.zeros((len(indices), 1))
                        alps = _alps(where, rate_mat, species, alp_mat, indices)
                        tau = _refresh_tau(ran_tor, step_cells[where], where, alps)
                        epoch = epoch_mat[step_cells[where]-1, where] + tau
                        stream_epoch, stream_alps = _streamer_put(stream_epoch, stream_alps, epoch[0], where, alps[:, 0])
                        ####
                    ## Jump Diffuse [Final]
                    temp = pre[:, cell] + delta
                    state_tor[0, :, cell] = temp
                    state_tor_objective[step_cell, :, cell] = temp[species_objective_indices]
                    # Alpha [Final]
                    # Cushion [Start]
                    step_cells[cell] += 1
                    if np.any(step_cells >= steps):
                        # self.step_cells = step_cells
                        break
                    pre[:, cell] = temp
                    # Cushion [Final]
                    # Beta [Start]
                    step_cell = step_cells[cell]
                    _species = pre[:, cell].copy()
                    species = _species.reshape((-1, 1))
                    alp_mat = np.zeros((len(indices), 1))
                    alps = _alps(cell, rate_mat, species, alp_mat, indices)
                    tau = _refresh_tau(ran_tor, step_cell, cell, alps)
                    epoch = epoch_mat[step_cell-1, cell] + tau
                    stream_epoch, stream_alps = _streamer_put(stream_epoch, stream_alps, epoch[0], cell, alps[:, 0])
                    # Beta [Final]
                    # Halt Flag [Start]
                    if halt_flag:
                        break
                    # Halt Flag [Final]
                return epoch_mat, state_tor, state_tor_objective, step_cells
        return __meth_direct__
    
    def meth_direct(self, jump_diffuse_tor = None, epoch_halt_tup = (None, None)):
        #
        global prop_funs
        global _alps
        global _refresh_tau
        global _refresh_delta
        # global _refresh_rate_mat
        global _streamer_make
        global _streamer_put
        global _streamer_get
        global _meth_direct
        #
        prop_funs = self.stem.prop_funs
        _alps = self._alps()
        _refresh_tau = self._refresh_tau()
        _refresh_delta = self._refresh_delta()
        # _refresh_rate_mat = self._refresh_rate_mat()
        _streamer_make = self._streamer('make')
        _streamer_put = self._streamer('put')
        _streamer_get = self._streamer('get')
        _meth_direct = self._meth_direct()
        #
        self._ran_tor()
        self._epoch_mat()
        self._state_tor()
        self._rate_mat()
        #
        ran_tor = self.ran_tor
        epoch_mat = self.epoch_mat
        state_tor = self.state_tor
        state_tor_objective = self.state_tor_objective
        rate_mat = self.rate_mat # rates ~ rate_mat # rates = np.array(list(stem.rates.values()))
        #
        stem = self.stem
        alp_mat = np.zeros((len(stem.assembly['prop_funs']), self.cells))
        indices = np.arange(len(stem.assembly['prop_funs']))
        delta_mat = stem.assembly['delta_mat']
        #
        pre = self._pre().copy()
        step_cells = np.array([1]*self.cells)
        species_objective_indices = np.array(self.species_objective_indices)
        # Step Zero [Start]
        step = 0
        cell = ...
        species = state_tor[step, :, cell]
        alps = _alps(0, rate_mat, species, alp_mat, indices) # Careful! Cell ~ Integer!
        tau = _refresh_tau(ran_tor, step+1, cell, alps) # Next Step! Step 1 Tau!
        stream_epoch, stream_alps = _streamer_make(self.cells, len(stem.assembly['prop_funs']))
        for cell in range(self.cells):
            stream_epoch, stream_alps = _streamer_put(stream_epoch, stream_alps, tau[cell], cell, alps[:, cell])
        # Step Zero [Final]
        # Jump Diffuse [Start]
        jump_diffuse_vet = self.stem.assembly['jump_diffuse_vet']
        if jump_diffuse_tor is None:
            jump_diffuse_comm_classes = None
        else:
            jump_diffuse_comm_classes = self.jump_diffuse_comm_classes
        jump_diffuse_delta_mat = self.stem.assembly['jump_diffuse_delta_mat']
        # Jump Diffuse [Final]
        # Epoch Halt [Start]
        noms = {'Seconds': 0, 'Minutes': 1, 'Hours': 2}
        _nom = noms[epoch_halt_tup[1]]
        nom = np.power(60, _nom)
        epoch_halt = epoch_halt_tup[0] * nom
        # Epoch Halt [Final]
        #
        self.epoch_mat, self.state_tor, self.state_tor_objective, self.step_cells = _meth_direct(self.steps, self.cells, step_cells, stream_epoch, stream_alps, epoch_mat, pre, state_tor, state_tor_objective, species_objective_indices, ran_tor, rate_mat, indices, delta_mat, jump_diffuse_vet, jump_diffuse_comm_classes, jump_diffuse_tor, jump_diffuse_delta_mat, epoch_halt)
        #
        return self
