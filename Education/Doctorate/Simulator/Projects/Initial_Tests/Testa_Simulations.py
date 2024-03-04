#%%#

# Do!
steps = 200000
trajectories = 1
w = BiochemSimul(e, steps, trajectories)
w.meth_direct()

# Analize
alias = BiochemAnalysis(w)
what = 'nor' # 'hist'
where = (0, 200000) # Time slicing
trajectory = 0
species = 'mRNA'

# Stats!
alias.plotful(what, where, species, trajectory)
alias.equi_stats(species, trajectory, 100)
alias.mean(species, trajectory) # Also add slicing
alias.variance(species, trajectory)

#%%#

# nP = 4
NTF = 2000
steps = 200000 # steps = 50000
trajectories = 1
jumps = 100
species = 'mRNA'

pre = 2000 # 2000 # 20
rates = {'kf': 0.01, 'kd': pre*0.0017, 'km': pre*0.17/1} # rates = {'kf': 0.5, 'kd': 0.005}
rates.update({f'kb{_}': (k0 if _ == 1 else k0*pow(1/u, _-1)) for _ in range(1, nP+1)})

d = simulator(nP, NTF, rates, steps, trajectories, jumps, species)
print(d)
exec(f'd{pre} = d')

#%%#

def showbiz(d, pre, species, arra, save = False):
    
    _NTF = list(d.keys())
    Ave = np.array([np.mean(d[i]['Ave']) for i in _NTF])
    Std = np.sqrt(np.array([np.mean(d[i]['Var']) for i in _NTF]))
    Ave[0] = 0
    Std[0] = 0
    
    if arra in (-1, 0):
        plt.plot(_NTF, Ave, color = 'green', marker = '+')
        plt.plot(_NTF, Ave + Std, color = 'pink', marker = '.')
        plt.plot(_NTF, Ave - Std, color = 'pink', marker = '.')
        plt.title(f'{pre}\nnTF ~ Mean (-|+ Std)\n{species}')
        if save:
            plt.savefig(f'{species}_nTF_Mean(-+Std)_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    if arra in (-1, 1):
        plt.plot(_NTF, Std / Ave, marker = '+')
        plt.title(f'{pre}\nnTF ~ Std/Mean\n{species}')
        if save:
            plt.savefig(f'{species}_CV_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    if arra in (-1, 2):
        plt.plot(_NTF, np.power(Std, 2) / Ave, marker = '+')
        plt.hlines(0.1, 0, NTF)
        plt.title(f'{pre}\nnTF ~ Variance/Mean\n{species}')
        if save:
            plt.savefig(f'{species}_Fan_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    if arra in (-1, 3):
        plt.plot(_NTF, Std, marker = '+')
        plt.plot(_NTF, np.power(Std, 2), marker = '+')
        plt.title(f'{pre}\nnTF ~ Std\nnTF ~ Variance\n{species}')
        if save:
            plt.savefig(f'{species}_NTF_Std_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    if arra in (-1, 4):
        plt.plot(Ave, Std)
        plt.plot(Ave, np.power(Ave, 1/2))
        plt.title(f'{pre}\nMean ~ Std\nMean ~ sqrt(Mean)\n{species}')
        if save:
            plt.savefig(f'{species}_Ave_(Std_OR_sqrt(Ave))_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    return None

#%%#

def _showbiz(d, pre, species, arra, save = False):
    
    ds = (20, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
    
    light = tuple(matplotlib.colors.TABLEAU_COLORS.keys())
    dark = ('darkblue', 'darkorange', 'darkgreen', 'darkred', 'violet', 'maroon', 'hotpink', 'darkgray', 'darkolivegreen', 'darkturquoise')
    
    _NTF = list(d.keys())
    Ave = np.array([np.mean(d[i]['Ave']) for i in _NTF])
    Std = np.sqrt(np.array([np.mean(d[i]['Var']) for i in _NTF]))
    Ave[0] = 0
    Std[0] = 0
    
    if arra in (-1, 0):
        plt.plot(_NTF, Ave, color = 'green', alpha = 1-ds.index(pre)*0.1, marker = '+')
        plt.plot(_NTF, Ave + Std, color = 'hotpink', alpha = 1-ds.index(pre)*0.1, marker = '.')
        plt.plot(_NTF, Ave - Std, color = 'pink', alpha = 1-ds.index(pre)*0.1, marker = '.')
        plt.title(f'nTF ~ Mean (-|+ Std)\n{species}')
        if save and pre == 2000:
            plt.savefig(f'{species}_nTF_Mean(-+Std)_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    if arra in (-1, 1):
        plt.plot(_NTF, Std / Ave, marker = '+')
        plt.title(f'nTF ~ Std/Mean\n{species}')
        if save:
            plt.savefig(f'{species}_CV_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    if arra in (-1, 2):
        plt.plot(_NTF, np.power(Std, 2) / Ave, marker = '+')
        plt.hlines(0.1, 0, NTF)
        plt.title(f'nTF ~ Variance/Mean\n{species}')
        if save:
            plt.savefig(f'{species}_Fan_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    if arra in (-1, 3):
        plt.plot(_NTF, Std, marker = '+')
        plt.plot(_NTF, np.power(Std, 2), marker = '+')
        plt.title(f'nTF ~ Std\nnTF ~ Variance\n{species}')
        if save:
            plt.savefig(f'{species}_NTF_Std_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    if arra in (-1, 4):
        plt.plot(Ave, Std)
        plt.plot(Ave, np.power(Ave, 1/2), color = 'black')
        plt.title(f'Mean ~ Std\nMean ~ sqrt(Mean)\n{species}')
        if arra == -1:
            plt.show()
        if save and pre == 2000:
            plt.savefig(f'{species}_Ave_(Std_OR_sqrt(Ave))_{pre}.jpeg', dpi = 250, quality = 95)
    
    return None

#%%# This cell stores/dumps the data to a "pickle" file!

file = open(f'd{pre}', 'wb')
exec(f'pickle.dump(d{pre}, file)')
file.close()

# file = open(f'd{pre}', 'rb')
# exec(f'd{pre} = pickle.load(file)')
# file.close()

#%%# The following cells are set for plotting!

ds = (20, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)

#%%#

for _ in ds:
    print(f'd{_}')
    file = open(f'd{_}', 'rb')
    exec(f'd{_} = pickle.load(file)')
    file.close()

#%%#

for pre in ds: 
    print(pre)
    exec(f'd = d{pre}')
    arra = 4
    _showbiz(d, pre, 'mRNA', arra, False)
    if arra == -1:
        pass
    else:
        pass
        # plt.legend(labels = ds, loc = 0)

#%%#


