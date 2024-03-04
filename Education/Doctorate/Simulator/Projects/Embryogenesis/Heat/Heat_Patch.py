#%%# Load AND Patch Data [Info]

path = '/home/mars-fias/Downloads/Clue_Data/Heat_Data'
nuns = [1, 2]
for nun in nuns:
    zea = 250
    alp = 121
    bet = 100
    file = open(f'{path}/{nun}_{zea}_{alp}_{bet}_conus', 'rb')
    if nuns.index(nun) == 0:
        info = pickle.load(file)
    else:
        info.update(pickle.load(file))
    file.close()

#%%# Plot Info Intro

scope = list({_[0] for _ in info.keys()})
scope.sort()
simas = len({_[1] for _ in info.keys()})
cycles = list({_[2] for _ in info.keys()})

_info = {(key, key_0, key_1): None for key in scope for key_0 in range(simas) for key_1 in cycles}
stars = np.full((len(scope), len(cycles)), np.nan, 'O')

def loss(w):
    ex = 2
    t = np.sum(w)
    n = w[1]
    g = w[2]
    x = np.power(n+g, ex)/np.power(t, ex)
    y = np.power(np.abs(n-g), ex)/np.power(t, ex)
    z = x-y
    return z

# Calculate my loss!
for mules in scope:
    for sima in range(simas):
        for cycle in cycles:
            temp = info[(mules, sima, cycle)]
            _info[(mules, sima, cycle)] = np.apply_along_axis(loss, 1, temp)

# Use my loss to create a heat map!
for mules in scope:
    for cycle in cycles:
        for sima in range(simas):
            temp = _info[(mules, sima, cycle)]
            if sima == 0:
                extra = temp
            else:
                extra = np.vstack((extra, temp))
        stars[scope.index(mules), cycle] = extra.mean(0) # Adapter!

#%%# Plot Info

import seaborn as sns

_scope_alp = np.sort(np.array(list({_[0] for _ in scope})))
_scope_bet = np.sort(np.array(list({_[1] for _ in scope})))
Y = _scope_alp
X = _scope_bet

plt.rcParams['figure.figsize'] = (2*7, 7) # (5, 5)
# link = https://matplotlib.org/stable/tutorials/colors/colormaps.html
cap = 1

safe = False
testa = '(cycle+1 == 4 and t == 12) or (cycle+1 == 5)'
epoch = 0
for cycle in cycles:
    for index in range(len(scope)):
        print(scope[index], cycle)
        if index == 0:
            temp = stars[index, cycle]
        else:
            temp = np.vstack((temp, stars[index, cycle]))
    for t in range(temp.shape[1]):
        Z = temp[:, t].reshape((len(Y), len(X))) # vmax = reps
        axe = sns.heatmap(data = Z, vmin = 0, vmax = 1, cmap = 'hot', annot = False, square = True, linewidth = 0, xticklabels = cap*X, yticklabels = cap*Y)
        axe.invert_yaxis()
        tit = f'Cycle = {cycle+1}\n# Cells = {np.power(2, cycle)}|{np.power(2, cycle-1) if cycle >= asymmetric else np.power(2, cycle)} | Inter-Division Time = {mates[cycle]} Hours\nTime = [{np.round(epoch/24, 1)}, {np.round((epoch + mates[cycle])/24, 1)}] Days\nHour = {t}'
        plt.title(tit)
        plt.xlabel('Activate')
        plt.ylabel('Repress')
        if safe and eval(testa):
            plt.savefig(f'{path}/Heat_{cycle+1}_{t}_Mean_True.jpeg')
        plt.show()
    epoch = epoch + mates[cycle]

#%%# Plot Info Extra

import seaborn as sns

_scope_alp = np.sort(np.array(list({_[0] for _ in scope})))
_scope_bet = np.sort(np.array(list({_[1] for _ in scope})))
Y = _scope_alp
X = _scope_bet

plt.rcParams['figure.figsize'] = (2*8, 7) # (2*5, 4)
# link = https://matplotlib.org/stable/tutorials/colors/colormaps.html
coloring = list(matplotlib.colors.TABLEAU_COLORS)
cap = 1

safe = False
testa = '(cycle+1 == 4 and t == 12) or (cycle+1 == 5)'
epoch = 0
for cycle in cycles:
    for index in range(len(scope)):
        print(scope[index], cycle)
        if index == 0:
            temp = stars[index, cycle]
        else:
            temp = np.vstack((temp, stars[index, cycle]))
    for t in range(temp.shape[1]):
        Z = temp[:, t].reshape((len(Y), len(X)))
        #
        fig, axe = plt.subplots(1, 1, constrained_layout = True)
        tit = f'Cycle = {cycle+1}\n# Cells = {np.power(2, cycle)}|{np.power(2, cycle-1) if cycle >= asymmetric else np.power(2, cycle)} | Inter-Division Time = {mates[cycle]} Hours\nTime = [{np.round(epoch/24, 1)}, {np.round((epoch + mates[cycle])/24, 1)}] Days\nHour = {t}'
        fig.suptitle(tit) # vmax = reps # levels = np.linspace(0, reps, int(2*reps/10+1))
        cor = axe.contourf(X, Y, Z, cmap = 'hot', vmin = 0, vmax = 1, levels = np.linspace(0, 1, int(2*reps/10+1)))
        # cor = axe.contourf(X, Y, Z, cmap = 'hot', vmin = 0, vmax = reps)
        car = fig.colorbar(mappable = cor, ax = axe, location = 'right')
        # car.ax.set_ylim(0, reps)
        axe.grid(color = coloring[7], linestyle = '--', alpha = 0.1)
        axe.set_xlabel('Activate')
        axe.set_xticks(X)
        axe.set_ylabel('Repress')
        axe.set_yticks(Y)
        if safe and eval(testa):
            plt.savefig(f'{path}/Cone_{cycle+1}_{t}_Mean.jpeg')
        plt.show()
        #
    epoch = epoch + mates[cycle]
