#%%# Definitions

# P_ := Promoter Binding Site _
# TF := Transcription Factor
# mRNA

nP = 4 # nP = 4 ~ nTF = 1000
nTF = 2000 # 940
initial_state = {f'P{_}': (1 if _ == 0 else 0) for _ in range(nP+1)}
initial_state.update({'TF': nTF, 'mRNA': 0})

# rates = {'kf': 0.1, 'kb': 1}
u = 5
k0 = 100
pre = 1 # 1000 # After to 20
kappa = 1 # (1, 5, 10, 15, 20)
rates = {'kf': 0.01, 'kd': pre*0.0017, 'km': pre*0.17/kappa} # rates = {'kf': 0.5, 'kd': 0.005}
rates.update({f'kb{_}': (k0 if _ == 1 else k0*pow(1/u, _-1)) for _ in range(1, nP+1)})

e = BiochemStem(initial_state, rates)

# Forward

for _ in range(nP):
    prop_fun = f'P{_}*TF*kf'
    delta = {f'P{_}': -1, f'P{_+1}': 1, 'TF': -1}
    e.add_reaction(f'P{_+1}f', prop_fun, delta)

e.add_reaction('mRNAf', f'P{nP}*km', {'mRNA': kappa})

# Backward

for _ in range(nP):
    prop_fun = f'P{_+1}*kb{_+1}'
    delta = {f'P{_}': 1, f'P{_+1}': -1, 'TF': 1}
    e.add_reaction(f'P{_}b', prop_fun, delta)

e.add_reaction('mRNAb', 'mRNA*kd', {'mRNA': -1})

# BiochemStem Final # Control Transcription

e.assemble()
e.assembly

#%%# This is just a preliminary analysis

# Do!
steps = 100000
trajectories = 1
w = BiochemSimul(e, steps, trajectories)
w.meth_direct()

#%%#

# Analize
alias = BiochemAnalysis(w)
what = 'nor' # 'hist'
where = (0, 10000) # Time slicing
trajectory = 0
species = 'mRNA'

# Stats!
alias.plotful(what, where, species, trajectory)








#%%# This is where the maigc should occur: we will define the simulation loop!

# Please, run the 'simulator' function definition: it is in 'Testa'!

# nP = 4
NTF = 2000
steps = 100000 # steps = 50000
trajectories = 1
jumps = 100
species = 'mRNA'

ds = (20, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)

for pre in ds:
    # pre = 20 # 2000 # 20
    print('\n\nMulti\t=\t', pre)
    kappa = 1 # (1, 5, 10, 15, 20)
    print('kappa\t=\t', kappa)
    rates = {'kf': 0.01, 'kd': pre*0.0017, 'km': pre*0.17/kappa} # rates = {'kf': 0.5, 'kd': 0.005}
    rates.update({f'kb{_}': (k0 if _ == 1 else k0*pow(1/u, _-1)) for _ in range(1, nP+1)})

    d = simulator(nP, NTF, rates, steps, trajectories, jumps, species, kappa)
    print(d)
    exec(f'd{pre} = d')
    
    # These lines of code stores/dumps the data to a "pickle" file!
    
    file = open(f'{kappa}/d{pre}', 'wb')
    exec(f'pickle.dump(d{pre}, file)')
    file.close()

#%%#

def _showbiz_(d, pre, species, arra, save = False):

    # ds = (20, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
    
    light = tuple(matplotlib.colors.TABLEAU_COLORS.keys())
    dark = ('darkblue', 'darkorange', 'darkgreen', 'darkred', 'violet', 'maroon', 'hotpink', 'darkgray', 'darkolivegreen', 'darkturquoise')
    
    _NTF = list(d.keys())
    Ave = np.array([np.mean(d[i]['Ave']) for i in _NTF])
    Std = np.sqrt(np.array([np.mean(d[i]['Var']) for i in _NTF]))
    Ave[0] = 0
    Std[0] = 0
    
    _cs = kaput.index(kappa) # 0.2+kaput.index(kappa)*0.2
    #_cs = ds.index(pre) # 1-ds.index(pre)*0.1 # It's not d, it is pre!
    cs = (light[_cs], dark[_cs])
    
    if arra in (-1, 0):
        plt.plot(_NTF, Ave, color = cs[0], alpha = 0.2+kaput.index(kappa)*0.2, marker = '+')
        plt.plot(_NTF, Ave + Std, color = cs[1], alpha = 0.2+kaput.index(kappa)*0.2, marker = '.')
        plt.plot(_NTF, Ave - Std, color = cs[1], alpha = 0.2+kaput.index(kappa)*0.2, marker = '.')
        plt.title(f'nTF ~ Mean (-|+ Std)\n{species}')
        plt.ylim(-30, 130)
        if save and pre == 2000 and kappa == 20:
            plt.savefig(f'{species}_nTF_Mean(-+Std)_{pre}_Many.jpeg', dpi = 250, quality = 95)
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
        plt.hlines(0.1, 0, _NTF)
        plt.title(f'nTF ~ Variance/Mean\n{species}')
        if save:
            plt.savefig(f'{species}_Fan_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    keys
    if arra in (-1, 3):
        plt.plot(_NTF, Std, marker = '+')
        plt.plot(_NTF, np.power(Std, 2), marker = '+')
        plt.title(f'nTF ~ Std\nnTF ~ Variance\n{species}')
        if save:
            plt.savefig(f'{species}_NTF_Std_{pre}.jpeg', dpi = 250, quality = 95)
        if arra == -1:
            plt.show()
    
    if arra in (-1, 4):
        plt.plot(Ave, np.power(Std, 1), color = cs[0], alpha = 0.2+kaput.index(kappa)*0.2)
        plt.plot(Ave, np.power(Ave, 1/2), color = 'black')
        plt.title(f'Mean ~ Std    Mean ~ sqrt(Mean)\n{species}')
        if arra == -1:
            plt.show()
        if save and pre == 2000 and kappa == 20:
            plt.savefig(f'{species}_Ave_(Std_OR_sqrt(Ave))_{pre}_Many.jpeg', dpi = 250, quality = 95)
    
    return None


#%%# This is the "copy" version! Be careful!

def _showbiz_(d, pre, species, arra, save = False):

    # ds = (20, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
    
    light = tuple(matplotlib.colors.TABLEAU_COLORS.keys())
    dark = ('darkblue', 'darkorange', 'darkgreen', 'darkred', 'violet', 'maroon', 'hotpink', 'darkgray', 'darkolivegreen', 'darkturquoise')
    
    _NTF = list(d.keys())
    Ave = np.array([np.mean(d[i]['Ave']) for i in _NTF])
    Std = np.sqrt(np.array([np.mean(d[i]['Var']) for i in _NTF]))
    Ave[0] = 0
    Std[0] = 0
    
    _cs = kaput.index(kappa) # 0.2+kaput.index(kappa)*0.2
    #_cs = ds.index(pre) # 1-ds.index(pre)*0.1 # It's not d, it is pre!
    cs = (light[_cs], dark[_cs])
    
    if arra in (-1, 0):
        plt.plot(_NTF, Ave, color = cs[0], alpha = 0.1+ds.index(pre)*0.1, marker = '+')
        plt.plot(_NTF, Ave + Std, color = cs[1], alpha = 0.1+ds.index(pre)*0.1, marker = '.')
        plt.plot(_NTF, Ave - Std, color = cs[1], alpha = 0.1+ds.index(pre)*0.1, marker = '.')
        plt.title(f'nTF ~ Mean (-|+ Std)\n{species}    {kappa}')
        plt.ylim(-30, 140)
        if save and pre == 2000:
            plt.savefig(f'{species}_nTF_Mean(-+Std)_{pre}_{kappa}.jpeg', dpi = 250, quality = 95)
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
        plt.hlines(0.1, 0, _NTF)
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
        plt.plot(Ave, np.power(Std, 2), color = cs[0], alpha = 0.1+kaput.index(kappa)*0.1)
        plt.plot(Ave, np.power(Ave, 1), color = 'black')
        plt.title(f'Mean ~ Variance\n{species}    {ds}    {kaput}')
        #plt.ylim(0, 55)
        if arra == -1:
            plt.show()
        if save and pre == 20 and kappa == 20:
            plt.savefig(f'{species}_Mean_Variance.jpeg', dpi = 250, quality = 95)
    
    return None

#%%# The following cells are set for plotting!

ds = (20, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
ds = 20,
kaput = (1, 5, 10, 15, 20)

for kappa in kaput:
    for _ in ds:
        print(f'd{_}')
        file = open(f'{kappa}/d{_}', 'rb')
        exec(f'k{kappa}_d{_} = pickle.load(file)')
        file.close()
    
    # One has to run first '_showbiz' fucntion: that's in 'Testa_Simulations'!
    
    for pre in ds:
        print(pre)
        exec(f'd = k{kappa}_d{pre}')
        arra = 4
        _showbiz_(d, pre, 'mRNA', arra, False)
        if arra == -1:
            pass
        else:
            pass
            # plt.legend(labels = ds, loc = 0)



#%%#

ds = (20, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
ds = 20,
kappa = 20

for _ in ds:
    print(f'd{_}')
    file = open(f'{kappa}/d{_}', 'rb')
    exec(f'k{kappa}_d{_} = pickle.load(file)')
    file.close()
    
# One has to run first '_showbiz' fucntion: that's in 'Testa_Simulations'!
    
for pre in ds:
    print(pre)
    exec(f'd = k{kappa}_d{pre}')
    arra = 4
    _showbiz_(d, pre, 'mRNA', arra, False)
    if arra == -1:
        pass
    else:
        pass
        # plt.legend(labels = ds, loc = 0)

#%%# This is for the new plots where we need specific points from each kappa, and a fixed pre.

# variance/mean burst
# Our null hypothesis still seems to be valid; namely, it looks like a linear model.
# We have a smoothing technique, which is some how wrong, but it's simple enough to give us some intuition.
# We did a sort of moving average.

from sklearn import datasets, linear_model

light = tuple(matplotlib.colors.TABLEAU_COLORS.keys())
dark = ('darkblue', 'darkorange', 'darkgreen', 'darkred', 'violet', 'maroon', 'hotpink', 'darkgray', 'darkolivegreen', 'darkturquoise')

ds = 20,
kaput = (1, 5, 10, 15, 20)
df = pd.DataFrame(columns = ['Burst', 'Mean', 'Variance', 'VM'])
df['Burst'] = kaput

take = 1

for kappa in kaput:
    for _ in ds:
        print(f'd{_}')
        file = open(f'{kappa}/d{_}', 'rb')
        temp = pickle.load(file)
        keys = list(temp.keys())
        selected = keys[len(keys)-take:]
        Mean = np.mean([temp[i]['Ave'] for i in selected])
        Variance = np.mean([temp[i]['Var'] for i in selected])
        df.loc[df.loc[:, 'Burst'] == kappa, 'Mean'] = Mean
        df.loc[df.loc[:, 'Burst'] == kappa, 'Variance'] = Variance
        df.loc[df.loc[:, 'Burst'] == kappa, 'VM'] = Variance/Mean
        file.close()

re = linear_model.LinearRegression()
X = np.array(df['Burst'])
X = X.reshape(-1, 1) # Puto sk-learn!
y = np.array(df['VM'])
re.fit(X = X, y = y)
predict = re.predict(X)

plt.bar(df.Burst, df.VM, color = light[0])
plt.plot(df.Burst, df.VM, color = dark[0])
plt.plot(df.Burst, predict, 'ro')
plt.plot(df.Burst, predict, color = 'red')
plt.ylim(0, 13)
plt.title(f'Burst Size ~ Variance/Mean\n{take}')

# This project is dead for now! We should organize it better!
