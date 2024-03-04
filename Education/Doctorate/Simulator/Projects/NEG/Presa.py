#%%# Keep this analysis cell [One]

def hist_trod(self, species, trajectories, threshold = None, path = None, safe = False, ret = False):
    
    _s = list(self.simul.stem.assembly['species'].values())
    s = _s.index(species)
    
    y = self.simul.state_tor[:, s, trajectories]
    times = np.diff(self.simul.epoch_mat[:, trajectories], axis = 0)
    maxi = np.sum(np.max(self.simul.epoch_mat[:, trajectories], axis = 0))
    
    elements = np.unique(y)
    ratios = np.zeros(shape = elements.shape)       
    
    it = np.nditer(op = [elements, ratios], op_flags = [['readonly'], ['writeonly']], order = 'C')
    
    with it:
        for e, r in it:
            summa = np.sum(np.where(y[:-1] == e, times, 0))
            r[...] = summa/maxi
    
    plt.plot(elements, ratios, drawstyle = 'steps-mid')
    plt.ylim(0, max(ratios)*(1.1))
    plt.title(species+'\nq = '+str(q))
    if safe:
        plt.savefig(path+'/'+str(q)+'_'+species+'_'+'EPDF'+'.jpeg', dpi = 250, quality = 95)
    plt.show()
    
    plt.plot(elements, np.cumsum(ratios), color = 'orange', drawstyle = 'steps-mid')
    plt.hlines(0.5, min(elements), max(elements))
    plt.ylim(0, 1.1)
    plt.title(species+'\nq = '+str(q))
    if safe:
        plt.savefig(path+'/'+str(q)+'_'+species+'_'+'ECDF'+'.jpeg', dpi = 250, quality = 95)
    plt.show()
    
    return (elements, ratios) if ret else None

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

# %time

alias = BiochemAnalysis(press)
# species = ''
trajectories = list(range(1000)) # list(range(250)) # 50
threshold = None
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Presa'
safe = True
ret = True
temp = hist_trod(alias, 'N', trajectories, threshold, path, safe, ret)
temp = hist_trod(alias, 'G', trajectories, threshold, path, safe, ret)

#plt.plot(self.simul.epoch_mat[:, trajectories], self.simul.state_tor[:, [0, 1], trajectories])

# Test!
#alias.equidistant(species, trajectory, level)
#plt.hist(alias.iy, bins = np.unique(alias.iy).shape[0], density = True)

#%%# Keep this analysis cell [Two]

def trod(self, species, trajectories, thresholds, ret = False):
    
    _s = list(self.simul.stem.assembly['species'].values())
    s = _s.index(species)
        
    y = self.simul.state_tor[:, s, trajectories]
    times = self.simul.epoch_mat[:, trajectories]
    
    where = y == threshold
    switches = np.diff(times[where])
    
    plt.plot(elements, ratios, drawstyle = 'steps-mid')
    plt.ylim(0, max(ratios)*(1.05))
    plt.show()
    
    return (elements, ratios) if ret else None

%time temp = trod(alias, 'N', 0, ret = True)

#%%# This cell has code mixing One and Two

from scipy import signal

print(signal.find_peaks_cwt(temp[1], np.arange(1, 10)))

print(signal.find_peaks(temp[1], 0.01))

np.sum(temp[1][9:100]) # Careful!
np.argmin(np.abs(np.cumsum(temp[1])-0.5))

#%%# TAP

def tap(self, species, trajectories, threshold = 50, sepal = 0.75, path = None, safe = False):
    
    tit = str(species)+'\nq = '+str(q)+' | '+str(threshold)+' | '+str(sepal)
    cos = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    labs = ['High '+species[_] if _ <= len(species)-1 else 'Other' for _ in range(len(species)+1)]
    counter = np.zeros(len(species)+1)
    _s = list(self.simul.stem.assembly['species'].values())
    s = [_s.index(species[0]), _s.index(species[1])]
    times = np.diff(self.simul.epoch_mat[:, trajectories], axis = 0)
    maxi = np.max(self.simul.epoch_mat[:, trajectories], axis = 0)
    ratios = np.zeros((len(trajectories), len(s)))
    
    for k in range(len(s)):
        y = self.simul.state_tor[:, s[k], trajectories]
        summa = np.sum(np.where(y[:-1] >= threshold, times, 0), axis = 0) # Threshold
        ratios[:, k] = summa/maxi
    
    abs_diff = np.abs(np.diff(ratios, axis = 1)).reshape(-1)
    where_sepal = abs_diff >= sepal # Sepal
    arm_sepal = np.argwhere(where_sepal) # Extra info!
    nota_sepal = np.argwhere(np.invert(where_sepal)) # Extra info!
    _which_high = np.argmax(ratios, axis = 1)
    which_high = _which_high[where_sepal]
    
    for k in range(len(s)):
        counter[k] = np.count_nonzero(which_high == k)
    counter[len(counter)-1] = len(trajectories)-len(which_high)
    
    plt.plot(ratios[:, 0], color = cos[0], marker = 'o', linestyle = '')
    plt.plot(ratios[:, 1], color = cos[1], marker = 'o', linestyle = '')
    plt.ylim(-0.05, 1.05)
    plt.title(tit)
    if safe:
        plt.savefig(path+'/'+str(q)+'_'+species[0]+species[1]+'_'+'HA'+'.jpeg', dpi = 250, quality = 95)
    plt.show()
    
    plt.plot(abs_diff, color = cos[2], marker = 'o', linestyle = '')
    plt.axhline(sepal, 0, len(trajectories)-1, color = 'black')
    plt.ylim(-0.05, 1.05)
    plt.title(tit)
    if safe:
        plt.savefig(path+'/'+str(q)+'_'+species[0]+species[1]+'_'+'HB'+'.jpeg', dpi = 250, quality = 95)
    plt.show()
    
    plt.hist(abs_diff, bins = 100, density = True, color = cos[2])
    plt.axvline(sepal, 0, len(trajectories)-1, color = 'black')
    plt.title(tit)
    if safe:
        plt.savefig(path+'/'+str(q)+'_'+species[0]+species[1]+'_'+'HC'+'.jpeg', dpi = 250, quality = 95)
    plt.show()
    
    plt.scatter(ratios[:, 0], ratios[:, 1], color = cos[3])
    plt.xlabel(species[0])
    plt.ylabel(species[1])
    plt.ylim(-0.05, 1.05)
    plt.title(tit)
    if safe:
        plt.savefig(path+'/'+str(q)+'_'+species[0]+species[1]+'_'+'HD'+'.jpeg', dpi = 250, quality = 95)
    plt.show()
    
    plt.bar(x = range(len(counter)), height = 100*counter/len(trajectories), color = cos[0:3], tick_label = labs)
    plt.title(tit)
    if safe:
        plt.savefig(path+'/'+str(q)+'_'+species[0]+species[1]+'_'+'HE'+'.jpeg', dpi = 250, quality = 95)
    plt.show()
    
    return (arm_sepal, nota_sepal, which_high, counter)

alias = BiochemAnalysis(press)

# %time

self = alias
species = ['N', 'G']
trajectories = list(range(1000)) # 50
threshold = 50
sepal = 0.75
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Presa'
safe = True
sieve = tap(self, species, trajectories, threshold, sepal, path, safe)

if safe:
    file = open(f'{path}/{q}_{species[0]}_{species[1]}_{len(trajectories)}_sieve', 'wb')
    exec(f'pickle.dump(sieve, file)')
    file.close()

#%%# TAPE

def expo(wind):
    if wind % 2 == 0:
        wind = wind + 1
    x = np.arange(wind)
    mu = x[int(wind/2)]
    sigma = np.sqrt(mu)
    z = np.exp(-np.power((x-mu)/sigma, 2)/2)/(np.power(2*np.pi*np.power(sigma, 2), 1/2))
    y = z/sum(z)
    return y

def tape(self, species, trajectories, sieve = None, threshold = 50, level = 0.001, kind = 0, wind = 10, tot = False, show = False, path = None, safe = False):
    
    tit = str(species)+'\nq = '+str(q)+' | '+str(threshold)+' | '+str(wind)
    cos = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    labs = None
    counter = np.full((2*len(species), len(trajectories)), np.nan)
    _s = list(self.simul.stem.assembly['species'].values())
    s = [_s.index(species[0]), _s.index(species[1])]
    times = self.simul.epoch_mat[:, trajectories]
    
    if sieve is None:
        print('All!')
    else:
        arm_sepal, nota_sepal, which_high, _counter = sieve
    
    for k in range(2*len(s)):
        if sieve is None:
            sieve_trajectories = trajectories
        else:
            if k < len(s):
                h = 0
                sieve_where = arm_sepal[which_high == k].reshape(-1)
            else:
                h = len(s)
                sieve_where = nota_sepal.reshape(-1)
            sieve_trajectories = np.array(trajectories)[sieve_where]
        for trajectory in sieve_trajectories:
            # Original data
            t = times[:, trajectory]
            y = self.simul.state_tor[:, s[k-h], trajectory]
            if show:
                plt.plot(t, y, color = cos[k])
                plt.title(tit + '\n' + str(trajectory))
                plt.show()
            # Downs data
            z = alias.equidistant(_s[s[k-h]], trajectory, level, kind)
            if show:
                plt.plot(z.ix, z.iy, color = cos[k])
                plt.title(tit)
                plt.show()
            # Con data
            wife = expo(wind)
            w = np.convolve(z.iy, wife)
            off = len(w)-len(wife)+1
            if show:
                plt.plot(z.ix, w[0:off], color = cos[k])
                plt.title(tit)
                plt.show()
            # Counter!
            con = np.count_nonzero(np.diff(np.sign(w[0:off]-threshold)))
            counter[k, trajectory] = con
            if show:
                print(_s[s[k-h]], '\t', trajectory, '\t', con)
    
    elements = np.unique(counter[np.invert(np.isnan(counter))].reshape(-1))
    swaps = np.full((2*len(s), len(elements)), np.nan)
    for k in range(2*len(s)):
        for e in range(len(elements)):
            swaps[k, e] = np.count_nonzero(counter[k] == elements[e])
    swaps[-1] = np.sum(swaps[len(s):2*len(s)], axis = 0)/2
    
    # Bar plot
    width = 0.25
    plt.bar(x = elements-width, height = 100*swaps[0]/len(trajectories), width = width, color = cos[0])
    plt.bar(x = elements, height = 100*swaps[1]/len(trajectories), width = width, color = cos[1], tick_label = elements)
    if tot:
        plt.bar(x = elements+width, height = 100*swaps[-1]/len(trajectories), width = width, color = cos[2])
    plt.xlabel('Number of switches')
    plt.ylabel('Trajectories %')
    plt.title(tit)
    if safe:
        plt.savefig(path+'/'+str(q)+'_'+species[0]+species[1]+'_'+'JA'+'_'+str(tot)+'.jpeg', dpi = 250, quality = 95)
    plt.show()
    
    return (swaps, elements, counter)

alias = BiochemAnalysis(press)

# %time

self = alias
species = ['N', 'G']
trajectories = list(range(1000)) # 50
sieve = sieve # tap
threshold = 50
level = 0.001
kind = 0
wind = 10
tot = True
show = False
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Presa'
safe = True
zoo = tape(self, species, trajectories, sieve, threshold, level, kind, wind, tot, show, path, safe)

if safe:
    file = open(f'{path}/{q}_{species[0]}_{species[1]}_{len(trajectories)}_zoo', 'wb')
    exec(f'pickle.dump(zoo, file)')
    file.close()

tot = False
tape(self, species, trajectories, sieve, threshold, level, kind, wind, tot, show, path, safe)

#%%# Extract!

path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Presa'
q = 1
species = ['N', 'G']
trajectories = list(range(1000))

file = open(f'{path}/{q}_{species[0]}_{species[1]}_{len(trajectories)}_sieve', 'rb')
exec(f'_sieve = pickle.load(file)')
file.close()

file = open(f'{path}/{q}_{species[0]}_{species[1]}_{len(trajectories)}_zoo', 'rb')
exec(f'_zoo = pickle.load(file)')
file.close()

#%%# Anime [Demo!]

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
h = pow(60, 2)
d = 24
k = 7

a = np.max(press.epoch_mat, axis = 0)/h/d
plt.plot(a)
plt.show()
s = [0, 1]
j = 50
t = press.epoch_mat[:, j]
z = press.state_tor[:, s, j]
plt.plot(t[t <= k*d*h]/h/d, z[t <= k*d*h, :])
plt.show()

#%%# Anime Data

self = press # Simulation
species = ['N', 'G']
s = [0, 1]
show = False
teds = [pow(60, 2), 24, 5] # teds = [pow(60, 2), 24, 7]
tie = np.linspace(0, int(teds[0]*teds[1]*teds[2]), int(teds[0]*teds[1]*teds[2])+1)
stamp = 4
ties = np.array([stamp*h for h in range(1, int(24/stamp)*teds[2]+stamp) if stamp*h <= teds[1]*teds[2]]) # Hours
sties = teds[0]*ties # Seconds
trajectories = range(self.state_tor.shape[2])
# trajectories = range(10)
conus = np.full((len(species), len(ties), len(trajectories)), np.nan)
maxi = np.max(self.state_tor[:, s, :])

for trajectory in trajectories:
    x = self.epoch_mat[:, trajectory]
    y = self.state_tor[:, s, trajectory]
    fun = interpolate.interp1d(x = x, y = y, kind = 0, axis = 0)
    z = fun(tie)
    conus[:, :, trajectory] = np.transpose(z[sties])
    if show:
        plt.plot(tie/teds[0]/teds[1], z)
        plt.plot(tie[sties]/teds[0]/teds[1], z[sties])
        plt.show()

#%%# Anime Plot [One]

safe = False
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Presa'
sepal = 25
cocos = np.linspace(0, 1, len(trajectories))
cam = matplotlib.cm.get_cmap('Spectral') # ['Spectral', 'RdYlGn', 'PiYG', 'coolwarm']

for t in range(len(ties)):
    tit = f'Q = {q} @ Time @ E{np.round(ties[t]/teds[1], 1)} = {ties[t]} Hours'
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, maxi)
    ax.set_ylim(0, maxi)
    ax.set_xlabel(species[0])
    ax.set_ylabel(species[1])
    ax.set_title(tit)
    x = conus[0, t, :]
    y = conus[1, t, :]
    ax.scatter(x, y, c = cocos, cmap = cam)
    ax.axhline(sepal, 0, maxi, color = 'gray')
    ax.axvline(sepal, 0, maxi, color = 'gray')
    ax.axline((0, 0), slope = 1, color = 'lightgray')
    if safe:
        plt.savefig(path+'/'+tit+'.jpeg', dpi = 250, quality = 95)
    plt.show()

#%%# Anime Plot [Two]

safe = False
path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/NEG/Presa'
sepal = 25
cos = ['tab:red', 'tab:olive', 'tab:cyan', 'tab:purple']
mares = ['.', '<', '>', '*']
mu = np.mean(conus, 2)
sigma = np.std(conus, 2)

for t in range(len(ties)):
    tit = f'S @ Q = {q} @ Time @ E{np.round(ties[t]/teds[1], 1)} = {ties[t]} Hours'
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, maxi)
    ax.set_ylim(0, maxi)
    ax.set_xlabel(species[0])
    ax.set_ylabel(species[1])
    ax.set_title(tit)
    x = mu[0, t]
    y = mu[1, t]
    ax.scatter(x, y, s = 200, color = cos[0], marker = mares[0])
    x = mu[0, t] + sigma[0, t]
    y = mu[1, t] + sigma[1, t]
    ax.scatter(x, y, s = 100, color = cos[1], marker = mares[1])
    x = np.max([0, mu[0, t] - sigma[0, t]])
    y = np.max([0, mu[1, t] - sigma[1, t]])
    ax.scatter(x, y, s = 100, color = cos[2], marker = mares[2])
    ax.axhline(sepal, 0, maxi, color = 'gray')
    ax.axvline(sepal, 0, maxi, color = 'gray')
    ax.axline((0, 0), slope = 1, color = 'lightgray')
    ax.grid(True, color = 'lavender', linestyle = 'dashed')
    if safe:
        plt.savefig(path+'/'+tit+'.jpeg', dpi = 250, quality = 95)
    plt.show()
