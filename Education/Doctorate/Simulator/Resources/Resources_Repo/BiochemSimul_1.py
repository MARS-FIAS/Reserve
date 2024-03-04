######## Name
######## BiochemSimul

######## Requires
######## {Modules}

class BiochemSimul:
    
    """Class 'BiochemSimul' Illustration!
    It is an essential part of the 'BiochemAnalysis' class!
    
    ########
    Attributes
        
    
    ########
    Methods
        
    
    """
    
    def __init__(self, stem, steps, trajectories, seed = None):
        """
        
        
        """
        if not stem._assembled:
            raise RuntimeError('The biochemical system is not assembled! Check it again!') # Stop!
        self.stem = stem
        self.steps = steps
        self.trajectories = trajectories
        self.seed = seed
    
    def __repr__(self):
        portrait = f'<{self.__class__.__name__}({self.stem.__class__.__name__}, {self.steps}, {self.trajectories})>'
        return portrait
    
    def __str__(self):
        portrait = repr(self)
        return portrait
    
    def _ran_tor(self):
        self.ran_tor = np.random.default_rng(seed = self.seed).uniform(size = (self.steps, 2, self.trajectories))
        self.ran_tor[:, 0, :] = -np.log(self.ran_tor[:, 0, :])
        return self
    
    def _epoch_mat(self):
        self.epoch_mat = np.full(shape = (self.steps, self.trajectories), fill_value = np.nan)
        self.epoch_mat[0] = 0
        return self
    
    def _state_tor(self):
        self.state_tor = np.zeros(shape = (self.steps, len(self.stem.assembly['species']), self.trajectories), dtype = np.uint32)
        return self
    
    def _iteration(self, t):
        self.state_tor[0, :, t] = list(self.stem.initial_state.values())
        it = np.nditer(op = self.state_tor[1:, :, t], flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = self.state_tor.shape[1])
        # it = np.nditer(op = self.state_tor[..., t], flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = self.state_tor.shape[1])
        return it
    
    def _tracker(self, it):
        _i = divmod(it.iterindex, self.state_tor.shape[1])
        i = _i[0] + 1
        return i
    
    def _pre(self, t):
        pre = self.state_tor[0, :, t]
        return pre
    
    def _alps(self, step, trajectory):
        for i, j in self.stem.rates.items():
            express = f'{i} = {j}'
            exec(express)
        for _ in self.stem.assembly['species']:
            key = self.stem.assembly['species'][_]
            value = self.state_tor[step - 1, _, trajectory]
            express = f'{key} = {value}'
            exec(express)
        _alps = []
        for alp in self.stem.assembly['prop_funs']:
            _alps.append(eval(alp))
        alps = np.zeros(len(_alps))
        alps[0] = np.sum(_alps)
        alps[1:] = np.cumsum(_alps[1:]) / alps[0]
        return alps
    
    def _refresh_epoch(self, step, trajectory, alps):
        tau = self.ran_tor[step, 0, trajectory] / alps[0]
        self.epoch_mat[step, trajectory] = self.epoch_mat[step - 1, trajectory] + tau
        return self
    
    def _refresh_delta(self, step, trajectory, alps):
        temp = np.argwhere(self.ran_tor[step, 1, trajectory] < alps[1:])
        if not any(temp):
            disc = len(alps) - 1
        else:
            disc = temp[0, 0] + 1
        delta = self.stem.assembly['delta_mat'][disc, :]
        return delta
    
    def meth_direct(self):
        self._ran_tor()
        self._epoch_mat()
        self._state_tor()
        for trajectory in range(self.trajectories):
            state_mat = self._iteration(trajectory)
            pre = self._pre(trajectory)
            for state in state_mat:
                step = self._tracker(state_mat) # [1, self.steps - 1]
                alps = self._alps(step, trajectory)
                self._refresh_epoch(step, trajectory, alps)
                delta = self._refresh_delta(step, trajectory, alps)
                state[...] = pre + delta
                pre = state
        return self

######## Name
######## BiochemSimulMule

######## Requires
######## {Modules}

class BiochemSimulMule:
    
    """Class 'BiochemSimul' Illustration!
    It is an essential part of the 'BiochemAnalysis' class!
    
    ########
    Attributes
        
    
    ########
    Methods
        
    
    """
    
    def __init__(self, stem, steps, cells, seed = None):
        """
        
        
        """
        if not stem._assembled:
            raise RuntimeError('The biochemical system is not assembled! Check it again!') # Stop!
        self.stem = stem
        self.steps = steps
        self.cells = cells
        self.seed = seed
    
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
        initial_state = np.array(list(self.stem.initial_state.values())).reshape(-1, 1)
        self.state_tor[0, ...] = np.repeat(initial_state, self.cells, 1)
        return self
    
    def _pre(self):
        pre = self.state_tor[0, ...]
        return pre
    
    def _alps(self, step, cell):
        for i, j in self.stem.rates.items():
            express = f'{i} = {j}'
            exec(express)
        if step == 0:
            for _ in self.stem.assembly['species']:
                key = self.stem.assembly['species'][_]
                value = self.state_tor[step, _, :]
                express = f'{key} = value'
                exec(express)
            alp_mat = np.zeros((len(self.stem.assembly['prop_funs']), self.cells))
            it = np.nditer(op = alp_mat, flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = self.cells)
            for alp in it:
                tracker = divmod(it.iterindex, self.cells)[0]
                prop_fun = self.stem.assembly['prop_funs'][tracker]
                alp[...] = eval(prop_fun)
            alp_mat[0] = np.sum(alp_mat, axis = 0)
            alp_mat[1:] = np.cumsum(alp_mat[1:], axis = 0) / alp_mat[0]
        else:
            for _ in self.stem.assembly['species']:
                key = self.stem.assembly['species'][_]
                value = self.state_tor[step, _, cell]
                express = f'{key} = value'
                exec(express)
            alp_mat = np.zeros((len(self.stem.assembly['prop_funs'])))
            it = np.nditer(op = alp_mat, flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = 1)
            for alp in it:
                tracker = it.iterindex
                prop_fun = self.stem.assembly['prop_funs'][tracker]
                alp[...] = eval(prop_fun)
            alp_mat[0] = np.sum(alp_mat, axis = 0)
            alp_mat[1:] = np.cumsum(alp_mat[1:], axis = 0) / alp_mat[0]
        return alp_mat
    
    def _refresh_tau(self, step, cell, alps):
        tau = self.ran_tor[step, 0, cell] / alps[0]
        return tau

    def _refresh_delta(self, step, cell, alps):
        clues_boa = self.ran_tor[step, 1, cell] <= alps[1:]
        _clues = np.indices(clues_boa.shape, dtype = np.int32)
        clues = _clues[0]
        temp = np.where(clues_boa, clues, 10*clues.shape[0])
        inventory = np.nanmin(temp, 0) # list(np.nanmin(temp, 0))
        delta_trap = self.stem.assembly['delta_mat'][1:]
        delta = np.transpose(delta_trap[inventory])
        return delta, inventory
    
    def meth_direct(self, jump_diffuse_mat = None):
        self._ran_tor()
        self._epoch_mat()
        self._state_tor()
        pre = self._pre().copy()
        stream = PriorityQueue()
        step_cells = np.array([1]*self.cells)
        # Step Zero [Start]
        step = 0
        cell = ...
        alps = self._alps(step, cell)
        tau = self._refresh_tau(step+1, cell, alps) # Step 1 Tau!
        for cell, epoch in enumerate(tau):
            stream.put((epoch, cell, alps[:, cell]))
        # Step Zero [Final]
        # Jump Diffuse [Start]
        jump_diffuse_vet = self.stem.assembly['jump_diffuse_vet']
        jump_diffuse_delta = self.stem.assembly['jump_diffuse_delta']
        # Jump Diffuse [Final]
        for step in range(1, self.steps*self.cells):
            # Alpha [Start]
            epoch, cell, alps = stream.get()
            sept = step_cells[cell]
            self.epoch_mat[sept, cell] = epoch
            delta, inventory = self._refresh_delta(sept, cell, alps)
            ## Jump Diffuse [Start]
            if jump_diffuse_vet[inventory] and jump_diffuse_mat is not None:
                ####
                if jump_diffuse_mat.size == self.cells:
                    jump_diffuse_where = jump_diffuse_mat
                else:
                    jump_diffuse_where = jump_diffuse_mat[step]
                ####
                self.epoch_mat[step_cells[jump_diffuse_where[cell]], jump_diffuse_where[cell]] = epoch
                temp = pre[:, jump_diffuse_where[cell]] + jump_diffuse_delta
                self.state_tor[step_cells[jump_diffuse_where[cell]], :, jump_diffuse_where[cell]] = temp
                ####
                step_cells[jump_diffuse_where[cell]] += 1
                if np.any(step_cells >= self.steps):
                    self.step_cells = step_cells
                    break
                pre[:, jump_diffuse_where[cell]] = temp
                ####
                ####
                _stream = stream.queue.copy()
                _dele = [_[1] for _ in stream.queue]
                dele = _dele.index(jump_diffuse_where[cell])
                _stream.pop(dele)
                stream = PriorityQueue()
                for _ in _stream:
                    stream.put(_)
                ####
                ####
                alps = self._alps(step_cells[jump_diffuse_where[cell]]-1, jump_diffuse_where[cell])
                tau = self._refresh_tau(step_cells[jump_diffuse_where[cell]], jump_diffuse_where[cell], alps)
                epoch = self.epoch_mat[step_cells[jump_diffuse_where[cell]]-1, jump_diffuse_where[cell]] + tau
                stream.put((epoch, jump_diffuse_where[cell], alps))
                ####
            ## Jump Diffuse [Final]
            temp = pre[:, cell] + delta
            self.state_tor[sept, :, cell] = temp
            # Alpha [Final]
            # Cushion [Start]
            step_cells[cell] += 1
            if np.any(step_cells >= self.steps):
                self.step_cells = step_cells
                break
            pre[:, cell] = temp
            # Cushion [Final]
            # Beta [Start]
            sept = step_cells[cell]
            alps = self._alps(sept-1, cell)
            tau = self._refresh_tau(sept, cell, alps)
            epoch = self.epoch_mat[sept-1, cell] + tau
            stream.put((epoch, cell, alps))
            # Beta [Final]
        return self
