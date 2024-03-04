#%%# Zero

def BiochemStemFun(radius, pol, _MRNA = 250, _protein = 4, mules = (0.25, 0.25)):
    
    # Biochemical System Definition
    
    # Regulation Motifs
    
    regulations = {'N': {'N': 1, 'E': -1, 'G': -1}, 'E': {'N': 0, 'E': 0, 'G': 0}, 'G': {'N': -1, 'E': 1, 'G': 1}}
    
    # Regulation: positive XOR negative
    
    positives = {X: {Y: np.nan if regulations[X][Y] == 0 else not np.signbit(regulations[X][Y]) for Y in regulations[X].keys()} for X in regulations.keys()}
    negatives = {X: {Y: np.nan if regulations[X][Y] == 0 else np.signbit(regulations[X][Y]) for Y in regulations[X].keys()} for X in regulations.keys()}
    repressors = negatives
    
    # Species/States
    
    proteins = [X for X in regulations.keys() if not all([Y == 0 for Y in regulations[X].values()])]
    MRNAs = [protein + '_MRNA' for protein in proteins]
    promoters = proteins
    binding_sites = proteins
    coop = 4
    cooperativity = [str(_) for _ in range(coop+1)] # Cooperativities?
    promoter_binding_sites = [P+S+C for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S]]
    
    # Artificial Species
    
    cases = ['A', 'I'] # {'A': 'Activated', 'I': 'Inactivated'}
    arts = [P+_ for P in promoters for _ in cases if not all([not _ for _ in regulations[P].values()])]
    
    # Artificial Species [Promoter Switching "Swift"]
    
    swift_arts = [P+'A_Swift' for P in promoters]
    arts.extend(swift_arts)
    
    # Rate Constants
    
    D = 10e-12 # pow(microns, 2)/seconds # pow(length, 2)/time # Protein diffusion constant
    protein_cross_section = 10e-9 # Nanometers
    cell_radius = radius*1e-6 # 10 Micrometers
    cell_volume = 4*math.pi*pow(cell_radius, 3)/3 # pow(meters, 3)
    
    pub = 5 # This is only valid for backward_rates
    kf_promoter = 4*math.pi*protein_cross_section*D/cell_volume # 3*pow(10, -4)
    # kb_promoter = pow(10, 3)*kf_promoter # 1/pow(5, s) # s ~ cooperativity ~ number of binding sites
    q = 1 # Important variable!
    # hack = 250 # Half Activation
    hare = mules[0]*10*1000 # Half Repression
    print('\n\n\tQ = ' + str(q) + '\t|\tHalf-Repression Threshold = ' + str(int(hare/10)) + '\t|\tCell Volume = ' + str(int(np.round(cell_volume*10e17))) + '\n\n')
    kb_promoter = {art: mules[1]*10*250*kf_promoter if 'A' in art else hare*kf_promoter*q if 'N' in art else hare*kf_promoter/q for art in arts}
    
    _MRNA_lifetime = (2, 4, 6, 8) # (5, 6, 7) # Hours
    MRNA_lifetime = _MRNA_lifetime[2]*pow(60, 2) # Seconds
    MRNA_copy_number = _MRNA # Steady-state (old) assumption # kf/kb = 1000
    
    share = 0.8 # (0, 1)
    kf_MRNA = share*MRNA_copy_number/MRNA_lifetime
    kb_MRNA = 1/MRNA_lifetime
    kf_spontaneous = (1-share)*MRNA_copy_number/MRNA_lifetime # kf_protein/10 # 10*kb_protein # mRNA leakage
    kb_spontaneous = kb_MRNA
    
    _protein_lifetime = (2, 4, 6, 8) # (5, 6, 7) # Hours
    protein_lifetime = _protein_lifetime[pol]*pow(60, 2) # Seconds
    protein_copy_number = _protein # Steady-state (old) assumption # kf/kb = 1000
    
    kf_protein = protein_copy_number/protein_lifetime
    kb_protein = 1/protein_lifetime
    
    # Biochemical System Construction
    
    # Spontaneous Reactions
    
    # Forward
    
    soft_exes = [f'0 -> {P}_MRNA' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    soft_props = [f'(1-np.sign({P}I))*kf_{P}_MRNA0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    soft_deltas = [{f'{P}_MRNA': 1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
    soft_rates = {f'kf_{P}_MRNA0': kf_spontaneous for P in proteins if not all([not _ for _ in regulations[P].values()])}
    soft_species = {f'{P}_MRNA': 0 for P in proteins if not all([not _ for _ in regulations[P].values()])}
    
    # Backward
    
    suba_exes = [f'{P}_MRNA -> 0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    suba_props = [f'{P}_MRNA*kb_{P}_MRNA0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    suba_deltas = [{f'{P}_MRNA': -1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
    suba_rates = {f'kb_{P}_MRNA0': kb_spontaneous for P in proteins if not all([not _ for _ in regulations[P].values()])}
    suba_species = soft_species
    
    # Non-Spontaneous Reactions # Function!
    
    # Forward
    
    forward_exes = [f'{S} + {P}{S}{C} -> {P}{S}'+str(int(C)+1) for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
    forward_props = [f'{P}{S}{C}*{S}*kf_{P}{S}'+str(int(C)+1) for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
    forward_deltas = [{f'{S}': -1, f'{P}{S}{C}': -1, f'{P}{S}'+str(int(C)+1): 1, P+'I' if repressors[P][S] else P+'A': 1 if int(C)+1 == coop else 0} for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
    forward_rates = {f'kf_{P}{S}'+str(int(C)+1): kf_promoter for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop}
    forward_species = {f'{P}{S}{C}': 1-np.sign(int(C)) for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S]}
    
    # Backward
    
    backward_exes = [f'{P}{S}'+str(int(C)+1)+f' -> {P}{S}{C} + {S}' for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
    backward_props = [f'{P}{S}'+str(int(C)+1)+f'*kb_{P}{S}{C}' for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
    backward_deltas = [{f'{S}': 1, f'{P}{S}{C}': 1, f'{P}{S}'+str(int(C)+1): -1, P+'I' if repressors[P][S] else P+'A': -1 if int(C)+1 == coop else 0} for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop]
    backward_rates = {f'kb_{P}{S}{C}': kb_promoter[f'{S}A']/pow(pub, int(C)) if positives[P][S] else kb_promoter[f'{S}I']/pow(pub, int(C)) for P in promoters for S in binding_sites for C in cooperativity if regulations[P][S] and int(C)+1 <= coop}
    backward_species = forward_species
    
    # Artificial Reactions
    
    art_exes = [f'{P}A -> {P}_MRNA' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    art_props = [f'np.sign({P}A)*(1-np.sign({P}I))*kf_{P}_MRNA1' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    art_deltas = [{f'{P}_MRNA': 1, f'{P}A_Swift': 1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
    art_rates = {f'kf_{P}_MRNA1': kf_MRNA for P in proteins if not all([not _ for _ in regulations[P].values()])}
    art_species = {art: 0 for art in arts}
    
    # Protein Reactions (Up)
    
    pro_up_exes = [f'{P}_MRNA -> {P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    pro_up_props = [f'{P}_MRNA*kf_{P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    pro_up_deltas = [{f'{P}': 1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
    pro_up_rates = {f'kf_{P}': kf_protein for P in proteins if not all([not _ for _ in regulations[P].values()])}
    pro_up_species = {P: 0 for P in proteins}
    
    # Protein Reactions (Down)
    
    pro_do_exes = [f'{P} -> 0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    pro_do_props = [f'{P}*kb_{P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    pro_do_deltas = [{f'{P}': -1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
    pro_do_rates = {f'kb_{P}': kb_protein for P in proteins if not all([not _ for _ in regulations[P].values()])}
    pro_do_species = pro_up_species
    
    # BiochemStem
    
    flags = ['soft', 'suba', 'forward', 'backward', 'art', 'pro_up', 'pro_do']
    
    initial_state = {}
    initial_state.update(soft_species)
    initial_state.update(forward_species)
    initial_state.update(art_species)
    initial_state.update(pro_up_species)
    
    rates = {}
    for _ in flags:
        exec(f'rates.update({_}_rates)')
    proto = BiochemStem(initial_state, rates)

    for flag in flags:
        indices = eval(f'range(len({flag}_exes))')
        for index in indices:
            name = eval(f'{flag}_exes[{index}]')
            prop_fun = eval(f'{flag}_props[{index}]')
            delta = eval(f'{flag}_deltas[{index}]')
            proto.add_reaction(name, prop_fun, delta, verbose = False)

    proto.assemble()
    # print(proto.assembly)
    
    # flag = 'forward'
    # for _ in ['exes', 'props', 'deltas', 'rates', 'species']:
    #     print('\n', eval(flag+'_'+_), '\n')
    
    return proto

#%%# BiochemSimul [Test One]

h = pow(60, 2)
d = 24
k = 5
s = ['N_MRNA', 'G_MRNA', 'N', 'G'] # [0, 1, 28, 29]

seed = 52 # 73
vomer = False
mate = k*d*h # Hours ~ Seconds
trajectories = 100
steps = 200000
instate = None
shutdowns = [] # [(0, 0.5), (7.5, 8)]

opts = [100/_ for _ in range(1, 101) if 100%_ == 0]
data = np.zeros((len(opts), len(s), trajectories, 2))

for index in range(len(opts)):
    _MRNA = opts[index]
    _protein = opts[::-1][index]
    print(f'MRNAs\t{_MRNA}\nProteins\t{_protein}')
    proto = BiochemStemFun(10, 1, _MRNA, _protein)
    press = BiochemSimulMuse(proto, instate, steps, trajectories, mate, vomer, shutdowns, seed)
    press.meth_direct()
    
    for j in range(trajectories):
        # j = 0
        t = press.epoch_mat[:, j]
        # z = press.state_tor[:, s, j]
        # plt.plot(t[t <= k*d*h]/h/d, z[t <= k*d*h, :])
        # plt.show()
        alias = BiochemAnalysis(press)
        where = (np.argmin(t < d*h), np.argmax(t > k*d*h))
        trajectory = j
        for species in s:
            # species = 'N'
            data[index, s.index(species), trajectory, 0] = alias.mean(where, species, trajectory)
            data[index, s.index(species), trajectory, 1] = alias.variance(where, species, trajectory)


#%%# BiochemAnalysis Zero

alp = np.zeros((len(opts), len(s), trajectories, 1))
alp[..., 0] = data[..., 1]/data[..., 0]
mean_data = np.nanmean(alp, 2)
star_data = np.nanstd(alp, 2)

plt.plot(mean_data[:, :, 0], label = s)
plt.legend()
plt.plot(star_data[:, :, 0], linestyle = ':')
plt.show()

plt.plot(mean_data[:, :, 0]+star_data[:, :, 0], linestyle = ':')
plt.plot(mean_data[:, :, 0]-star_data[:, :, 0], linestyle = ':')

j = 0
t = press.epoch_mat[:, j]
z = press.state_tor[:, [0, 1, 28, 29], j]
plt.plot(t[t <= k*d*h]/h/d, z[t <= k*d*h, :])
plt.show()

safe = False

if safe:
    plt.savefig(species+'_'+str(trajectory)+'.jpeg', dpi = 250, quality = 95)

#%%# BiochemAnalysis One

coloring = list(matplotlib.colors.TABLEAU_COLORS)

alp = np.zeros((len(opts), len(s), trajectories, 1))
alp[..., 0] = data[..., 1]/data[..., 0]
mean_data = np.nanmean(alp, 2)
star_data = np.nanstd(alp, 2)

for index in range(len(s)):
    plt.bar(x = range(len(opts)), height = mean_data[:, index, :].reshape(-1), tick_label = [f'{opts[index]}\n{opts[::-1][index]}' for index in range(len(opts))], color = coloring[index], label = f'{s[index]}\nM/P')
    plt.hlines(1, -1, len(opts), color = 'black')
    plt.legend()
    safe = False
    if safe:
        plt.savefig(s[index]+'.jpeg', dpi = 250)
    plt.show()
