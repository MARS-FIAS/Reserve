######## Name
######## BiochemStemFun

######## Requires
######## {Modules}

def BiochemStemFun(radius, pol):
    
    # Biochemical System Definition
    
    # Regulation Motifs
    
    regulations = {'N': {'N': 1, 'E': -1, 'G': -1}, 'E': {'N': 0, 'E': 0, 'G': 0}, 'G': {'N': -1, 'E': 1, 'G': 1}}
    
    # Regulation: positive XOR negative
    
    positives = {X: {Y: np.nan if regulations[X][Y] == 0 else not np.signbit(regulations[X][Y]) for Y in regulations[X].keys()} for X in regulations.keys()}
    negatives = {X: {Y: np.nan if regulations[X][Y] == 0 else np.signbit(regulations[X][Y]) for Y in regulations[X].keys()} for X in regulations.keys()}
    repressors = negatives
    
    # Species/States
    
    proteins = [X for X in regulations.keys() if not all([Y == 0 for Y in regulations[X].values()])]
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
    hare = 1000 # Half Repression
    print('\n\n\tQ = ' + str(q) + '\t|\tHalf-Repression Threshold = ' + str(int(hare/10)) + '\t|\tCell Volume = ' + str(int(np.round(cell_volume*10e17))) + '\n\n')
    kb_promoter = {art: 250*kf_promoter if 'A' in art else hare*kf_promoter*q if 'N' in art else hare*kf_promoter/q for art in arts}
    
    _protein_lifetime = (2, 4, 6, 8) # (5, 6, 7) # Hours
    protein_lifetime = _protein_lifetime[pol]*pow(60, 2) # Seconds
    protein_copy_number = 100 # Steady-state (old) assumption # kf/kb = 1000
    
    share = 0.8 # (0, 1)
    kf_protein = share*protein_copy_number/protein_lifetime
    kb_protein = 1/protein_lifetime
    kf_spontaneous = (1-share)*protein_copy_number/protein_lifetime # kf_protein/10 # 10*kb_protein # mRNA leakage
    kb_spontaneous = kb_protein
    
    # Biochemical System Construction
    
    # Spontaneous Reactions
    
    # Forward
    
    soft_exes = [f'0 -> {P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    soft_props = [f'(1-np.sign({P}I))*kf_{P}0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    soft_deltas = [{f'{P}': 1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
    soft_rates = {f'kf_{P}0': kf_spontaneous for P in proteins if not all([not _ for _ in regulations[P].values()])}
    soft_species = {f'{P}': 0 for P in proteins if not all([not _ for _ in regulations[P].values()])}
    
    # Backward
    
    suba_exes = [f'{P} -> 0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    suba_props = [f'{P}*kb_{P}0' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    suba_deltas = [{f'{P}': -1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
    suba_rates = {f'kb_{P}0': kb_spontaneous for P in proteins if not all([not _ for _ in regulations[P].values()])}
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
    
    art_exes = [f'{P}A -> {P}' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    art_props = [f'np.sign({P}A)*(1-np.sign({P}I))*kf_{P}1' for P in proteins if not all([not _ for _ in regulations[P].values()])]
    art_deltas = [{f'{P}': 1, f'{P}A_Swift': 1} for P in proteins if not all([not _ for _ in regulations[P].values()])]
    art_rates = {f'kf_{P}1': kf_protein for P in proteins if not all([not _ for _ in regulations[P].values()])}
    art_species = {art: 0 for art in arts}
    
    # BiochemStem
    
    flags = ['soft', 'suba', 'forward', 'backward', 'art']
    
    initial_state = {}
    initial_state.update(soft_species)
    initial_state.update(forward_species)
    initial_state.update(art_species)
    
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
    
    return proto
