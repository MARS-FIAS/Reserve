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
        self.state_tor = np.zeros(shape = (self.steps, len(self.stem.assembly['species']), self.trajectories), dtype = np.uint16)
        return self
    
    def _iteration(self):
        for t in range(self.state_tor.shape[2]):
            self.state_tor[0, :, t] = list(self.stem.initial_state.values())
        self.it = np.nditer(op = self.state_tor[1:, ...], flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = self.state_tor.shape[1]*self.state_tor.shape[2])
        # it = np.nditer(op = self.state_tor[..., t], flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = self.state_tor.shape[1])
        return self
    
    def _tracker(self):
        _i = divmod(self.it.iterindex, self.state_tor.shape[1]*self.state_tor.shape[2])
        i = _i[0] + 1
        return i
    
    def _pre(self):
        pre = self.state_tor[0, ...]
        return pre
    
    def _alps(self, step):
        for i, j in self.stem.rates.items():
            express = f'{i} = {j}'
            exec(express)
        for _ in self.stem.assembly['species']:
            key = self.stem.assembly['species'][_]
            value = self.state_tor[step - 1, _, :]
            express = f'{key} = value'
            exec(express)
        alp_mat = np.zeros((len(self.stem.assembly['prop_funs']), self.trajectories))
        it = np.nditer(op = alp_mat, flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = self.trajectories)
        for alp in it:
            tracker = divmod(it.iterindex, self.trajectories)[0]
            prop_fun = self.stem.assembly['prop_funs'][tracker]
            alp[...] = eval(prop_fun)
        alp_mat[0] = np.sum(alp_mat, axis = 0)
        alp_mat[1:] = np.cumsum(alp_mat[1:], axis = 0) / alp_mat[0]
        return alp_mat
    
    def _refresh_epoch(self, step, alps):
        tau = self.ran_tor[step, 0, :] / alps[0]
        self.epoch_mat[step, :] = self.epoch_mat[step - 1, :] + tau
        return self

    def _refresh_delta(self, step, alps):
        clues_boa = self.ran_tor[step, 1, :] <= alps[1:, :]
        _clues = np.indices(clues_boa.shape, dtype = np.int32)
        clues = _clues[0]
        temp = np.where(clues_boa, clues, 10*clues.shape[0])
        inventory = list(np.nanmin(temp, 0))
        delta_trap = self.stem.assembly['delta_mat'][1:]
        delta = np.transpose(delta_trap[inventory])
        return delta    
    
    def meth_direct(self):
        self._ran_tor()
        self._epoch_mat()
        self._state_tor()
        self._iteration()
        pre = self._pre()
        for state in self.it:
            step = self._tracker() # [1, self.steps - 1]
            alps = self._alps(step)
            self._refresh_epoch(step, alps)
            delta = self._refresh_delta(step, alps)
            pre_delta = np.ravel(pre) + np.ravel(delta)
            state[...] = pre_delta
            pre = state
        return self

######## Name
######## BiochemSimulMuse

######## Requires
######## {Modules}

class BiochemSimulMuse:
    
    """Class 'BiochemSimul' Illustration!
    It is an essential part of the 'BiochemAnalysis' class!
    
    ########
    Attributes
        
    
    ########
    Methods
        
    
    """
    
    def __init__(self, stem, instate, steps, trajectories, mate, vomer = False, shutdowns = list(), seed = None):
        """
        
        
        """
        if not stem._assembled:
            raise RuntimeError('The biochemical system is not assembled! Check it again!') # Stop!
        self.stem = stem
        self.instate = instate
        self.steps = steps
        self.trajectories = trajectories
        self.mate = mate
        self.vomer = vomer
        self.shutdowns = shutdowns
        self.seed = seed
        # Extra Attributes!
        self.mate_index = 0 # self.mate
        self.shutdowns_flag = any(self.shutdowns) # self.shutdowns
    
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
        self.state_tor = np.zeros(shape = (self.steps, len(self.stem.assembly['species']), self.trajectories), dtype = np.uint16)
        return self
    
    def _iteration(self):
        if self.instate is None:    
            for t in range(self.state_tor.shape[2]):
                self.state_tor[0, :, t] = list(self.stem.initial_state.values())
        else:
            self.state_tor[0, :, :] = self.instate
        self.it = np.nditer(op = self.state_tor[1:, ...], flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = self.state_tor.shape[1]*self.state_tor.shape[2])
        # it = np.nditer(op = self.state_tor[..., t], flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = self.state_tor.shape[1])
        return self
    
    def _tracker(self):
        _i = divmod(self.it.iterindex, self.state_tor.shape[1]*self.state_tor.shape[2])
        i = _i[0] + 1
        return i
    
    def _pre(self):
        pre = self.state_tor[0, ...]
        return pre
    
    def _alps(self, step):
        ########
        vomer = self.vomer
        mate = self.mate
        curt = np.min(self.epoch_mat[step-1, :]) # mate
        _mix = 'NG'
        mix = [alp+bet for alp in _mix for bet in _mix]
        ########
        shutdowns = self.shutdowns
        test = [False] * len(shutdowns)
        if self.shutdowns_flag:
            for shutdown in shutdowns:
                index = shutdowns.index(shutdown)
                test[index] = shutdown[0] <= curt <= shutdown[1]
            # print(curt/pow(60, 2), any(test))
        ########
        for i, j in self.stem.rates.items():
            shut_flag = any(test) and i in ['kf_'+alp+bet for alp in ('N', 'G') for bet in ('0', '1')]
            if vomer and any([_ in i for _ in mix]):
                express = f'{i} = {j/(1+curt/mate)}' if not shut_flag else f'{i} = {j/(100*(1+curt/mate))}' # Duplicate Volume =: K/2
            else:
                if any(test) and i in ['kf_'+alp+bet for alp in ('N', 'G') for bet in ('0', '1')]:
                    express = f'{i} = {j/100}'
                    # print(curt/pow(60, 2), i, j, j/10)
                else:
                    express = f'{i} = {j}'
            exec(express)
        for _ in self.stem.assembly['species']:
            key = self.stem.assembly['species'][_]
            value = self.state_tor[step - 1, _, :]
            express = f'{key} = value'
            exec(express)
        alp_mat = np.zeros((len(self.stem.assembly['prop_funs']), self.trajectories))
        it = np.nditer(op = alp_mat, flags = ['external_loop', 'buffered'], op_flags = ['readwrite'], order = 'C', buffersize = self.trajectories)
        for alp in it:
            tracker = divmod(it.iterindex, self.trajectories)[0]
            prop_fun = self.stem.assembly['prop_funs'][tracker]
            alp[...] = eval(prop_fun)
        alp_mat[0] = np.sum(alp_mat, axis = 0)
        alp_mat[1:] = np.cumsum(alp_mat[1:], axis = 0) / alp_mat[0]
        return alp_mat
    
    def _refresh_epoch(self, step, alps):
        tau = self.ran_tor[step, 0, :] / alps[0]
        self.epoch_mat[step, :] = self.epoch_mat[step - 1, :] + tau
        return self

    def _refresh_delta(self, step, alps):
        clues_boa = self.ran_tor[step, 1, :] <= alps[1:, :]
        _clues = np.indices(clues_boa.shape, dtype = np.int32)
        clues = _clues[0]
        temp = np.where(clues_boa, clues, 10*clues.shape[0])
        inventory = list(np.nanmin(temp, 0))
        delta_trap = self.stem.assembly['delta_mat'][1:]
        delta = np.transpose(delta_trap[inventory])
        return delta    
    
    def meth_direct(self):
        self._ran_tor()
        self._epoch_mat()
        self._state_tor()
        self._iteration()
        pre = self._pre()
        for state in self.it:
            step = self._tracker() # [1, self.steps - 1]
            alps = self._alps(step)
            self._refresh_epoch(step, alps)
            delta = self._refresh_delta(step, alps)
            pre_delta = np.ravel(pre) + np.ravel(delta)
            state[...] = pre_delta
            pre = state
            if np.all(self.epoch_mat[step, :] >= self.mate):
                self.mate_index = step
                break
        return self
