######## Name
######## BiochemSimulMuleRapid

######## Requires
######## {Modules}

class BiochemSimulMuleRapid:
    
    """Class 'BiochemSimul' Illustration!
    It is an essential part of the 'BiochemAnalysis' class!
    
    ########
    Attributes
        
    
    ########
    Methods
        
    
    """
    
    def __init__(self, stem, instate, steps, cells, seed = None, verbose = True):
        """
        
        
        """
        if not stem._assembled:
            raise RuntimeError('The biochemical system is not assembled! Check it again!') # Stop!
        self.stem = stem
        self.instate = instate
        self.steps = steps
        self.cells = cells
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
        self.state_tor = np.zeros(shape = (self.steps, len(self.stem.assembly['species']), self.cells), dtype = np.uint16)
        if self.instate is None:
            initial_state = np.array(list(self.stem.initial_state.values())).reshape(-1, 1)
            self.state_tor[0, ...] = np.repeat(initial_state, self.cells, 1)
        else:
            self.state_tor[0, ...] = self.instate
        return self
    
    def _pre(self):
        pre = self.state_tor[0, ...]
        return pre

    def _alps(self):
        if self.verbose:
            fun = '__alps__'
            arcs = ['rates', 'species', 'alp_mat', 'indices']
            print(fun, '\n\t', arcs)
        if '_alps' not in globals():
            global __alps__
            @numba.njit
            def __alps__(rates, species, alp_mat, indices):
                for cell in range(species.shape[1]):
                    _species = species[:, cell]
                    for index in indices:
                        alp_mat[index, cell] = prop_funs(index, rates, _species)
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
    
    def _meth_direct(self):
        if self.verbose:
            fun = '__meth_direct__'
            arcs = []
            print(fun, '\n\t', arcs)
        if '_meth_direct' not in globals():
            global __meth_direct__
            @numba.njit
            def __meth_direct__(steps, cells, step_cells, stream_epoch, stream_alps, epoch_mat, pre, state_tor, ran_tor, rates, indices, delta_mat, jump_diffuse_vet, jump_diffuse_mat, jump_diffuse_delta):
                for step in range(1, steps*cells):
                    # Alpha [Start]
                    stream_epoch, stream_alps, epoch, cell, alps = _streamer_get(stream_epoch, stream_alps)
                    sept = step_cells[cell]
                    epoch_mat[sept, cell] = epoch
                    delta, inventory = _refresh_delta(ran_tor, sept, cell, alps, delta_mat)
                    ## Jump Diffuse [Start]
                    if jump_diffuse_vet[inventory] and jump_diffuse_mat is not None:
                        ####
                        # if jump_diffuse_mat.size == cells:
                        #     jump_diffuse_where = np.copy(jump_diffuse_mat)
                        # else:
                        #     # jump_diffuse_where = np.array(jump_diffuse_mat[step])
                        #     pass
                        jump_diffuse_where = np.copy(jump_diffuse_mat[step])
                        ####
                        where = jump_diffuse_where[cell]
                        epoch_mat[step_cells[where], where] = epoch
                        temp = pre[:, where] + jump_diffuse_delta
                        state_tor[step_cells[where], :, where] = temp
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
                        _species = state_tor[step_cells[where]-1, :, where].copy()
                        species = _species.reshape((-1, 1))
                        alp_mat = np.zeros((len(indices), 1))
                        alps = _alps(rates, species, alp_mat, indices)
                        tau = _refresh_tau(ran_tor, step_cells[where], where, alps)
                        epoch = epoch_mat[step_cells[where]-1, where] + tau
                        stream_epoch, stream_alps = _streamer_put(stream_epoch, stream_alps, epoch[0], where, alps[:, 0])
                        ####
                    ## Jump Diffuse [Final]
                    temp = pre[:, cell] + delta
                    state_tor[sept, :, cell] = temp
                    # Alpha [Final]
                    # Cushion [Start]
                    step_cells[cell] += 1
                    if np.any(step_cells >= steps):
                        # self.step_cells = step_cells
                        break
                    pre[:, cell] = temp
                    # Cushion [Final]
                    # Beta [Start]
                    sept = step_cells[cell]
                    _species = state_tor[sept-1, :, cell].copy()
                    species = _species.reshape((-1, 1))
                    alp_mat = np.zeros((len(indices), 1))
                    alps = _alps(rates, species, alp_mat, indices)
                    tau = _refresh_tau(ran_tor, sept, cell, alps)
                    epoch = epoch_mat[sept-1, cell] + tau
                    stream_epoch, stream_alps = _streamer_put(stream_epoch, stream_alps, epoch[0], cell, alps[:, 0])
                    # Beta [Final]
                return epoch_mat, state_tor, step_cells
        return __meth_direct__
    
    def meth_direct(self, jump_diffuse_mat = None):
        #
        global prop_funs
        global _alps
        global _refresh_tau
        global _refresh_delta
        global _streamer_make
        global _streamer_put
        global _streamer_get
        global _meth_direct
        #
        prop_funs = self.stem.prop_funs
        _alps = self._alps()
        _refresh_tau = self._refresh_tau()
        _refresh_delta = self._refresh_delta()
        _streamer_make = self._streamer('make')
        _streamer_put = self._streamer('put')
        _streamer_get = self._streamer('get')
        _meth_direct = self._meth_direct()
        #
        self._ran_tor()
        self._epoch_mat()
        self._state_tor()
        #
        ran_tor = self.ran_tor
        epoch_mat = self.epoch_mat
        state_tor = self.state_tor
        #
        stem = self.stem
        rates = np.array(list(stem.rates.values()))
        alp_mat = np.zeros((len(stem.assembly['prop_funs']), self.cells))
        indices = np.arange(len(stem.assembly['prop_funs']))
        delta_mat = stem.assembly['delta_mat']
        #
        pre = self._pre().copy()
        step_cells = np.array([1]*self.cells)
        # Step Zero [Start]
        step = 0
        cell = ...
        species = state_tor[step, :, cell]
        alps = _alps(rates, species, alp_mat, indices)
        tau = _refresh_tau(ran_tor, step+1, cell, alps) # Next Step! Step 1 Tau!
        stream_epoch, stream_alps = _streamer_make(self.cells, len(stem.assembly['prop_funs']))
        for cell in range(self.cells):
            stream_epoch, stream_alps = _streamer_put(stream_epoch, stream_alps, tau[cell], cell, alps[:, cell])
        # Step Zero [Final]
        # Jump Diffuse [Start]
        jump_diffuse_vet = self.stem.assembly['jump_diffuse_vet']
        jump_diffuse_delta = self.stem.assembly['jump_diffuse_delta']
        # Jump Diffuse [Final]
        #
        self.epoch_mat, self.state_tor, self.step_cells = _meth_direct(self.steps, self.cells, step_cells, stream_epoch, stream_alps, epoch_mat, pre, state_tor, ran_tor, rates, indices, delta_mat, jump_diffuse_vet, jump_diffuse_mat, jump_diffuse_delta)
        #
        return self
