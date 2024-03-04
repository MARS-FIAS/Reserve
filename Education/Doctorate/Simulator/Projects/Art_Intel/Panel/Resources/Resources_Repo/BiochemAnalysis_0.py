######## Name
######## BiochemAnalysis

######## Requires
######## {Modules}

class BiochemAnalysis:
    
    """Class 'BiochemAnalysis' Illustration!
    It is an essential part of the 'BiochemBase' class!
    
    ########
    Attributes
        
    
    ########
    Methods
        
    
    """
    
    def __init__(self, simul):
        """
        
        
        """
        self.simul = simul
        self._equidistant = False
    
    def __repr__(self):
        portrait = f'<{self.__class__.__name__}({self.simul.__class__.__name__})>'
        return portrait
    
    def __str__(self):
        portrait = repr(self)
        return portrait
    
    def plotful(self, what, where, species, trajectory):
        
        _s = list(self.simul.stem.assembly['species'].values())
        s = _s.index(species)
        
        x = self.simul.epoch_mat[where[0]:where[1], trajectory]
        y = self.simul.state_tor[where[0]:where[1], s, trajectory]
        
        if what == 'nor':
            ave = self.mean(where, species, trajectory)
            std = np.sqrt(self.variance(where, species, trajectory))
            plt.plot(x, y, ls = '-', marker = '')
            plt.hlines([ave - std, ave, ave + std], 0, np.nanmax(x), colors = 'black', linestyles = 'dashed')
        elif what == 'hist':
            plt.hist(y, bins = np.unique(y).shape[0])
        else:
            print('Nothing to plot!')
        
        return None
    
    def equidistant(self, species, trajectory, level, kid):
        
        # Definitions!
        _s = list(self.simul.stem.assembly['species'].values())
        s = _s.index(species)
        
        x = self.simul.epoch_mat[:, trajectory]
        y = self.simul.state_tor[:, s, trajectory]

        ix = np.linspace(0, x[-1], int(level*len(x)))
        fun = interpolate.interp1d(x = x, y = y, kind = kid)
        iy = fun(ix)
        
        self.ix = ix
        self.iy = iy
        self._equidistant = True # plt.plot(_ix, _iy, ls = '-', marker = '+')
        
        return self
    
    def equi_stats(self, species, trajectory, level):
        self.equidistant(species, trajectory, level)
        stats = {'Mean': np.mean(self.iy), "Var": np.var(self.iy)}
        return stats
    
    # Stats!
    
    def mean(self, where, species, trajectory): # Only one trajectory for now!
        
        _s = list(self.simul.stem.assembly['species'].values())
        s = _s.index(species)
        
        y = self.simul.state_tor[where[0]:where[1], s, trajectory]
        elements = np.unique(y)
        times = np.diff(self.simul.epoch_mat[where[0]:where[1], trajectory])
        lis = []        
        for i in elements:
            _ = np.nansum(np.where(y[:-1] == i, times, 0))/np.nanmax(self.simul.epoch_mat[:, trajectory])
            lis.append(_)
        ave = np.average(a = elements, weights = np.array(lis))
        return ave
    
    def variance(self, where, species, trajectory): # Only one trajectory for now!
        
        _s = list(self.simul.stem.assembly['species'].values())
        s = _s.index(species)
        
        y = self.simul.state_tor[where[0]:where[1], s, trajectory]
        elements = np.unique(y)
        times = np.diff(self.simul.epoch_mat[where[0]:where[1], trajectory])
        lis = []        
        for i in elements:
            _ = np.nansum(np.where(y[:-1] == i, times, 0))/np.nanmax(self.simul.epoch_mat[:, trajectory])
            lis.append(_)
        ave = np.average(a = elements, weights = np.array(lis))
        var = np.average(a = np.power(elements, 2), weights = np.array(lis)) - pow(ave, 2)
        # print('Ave', ave, '\t', 'Var', var, '\t', 'Std', np.sqrt(var))
        return var
