############################
######## Grid_World ########
############################

#%%# Catalyzer [Next State AND (Next) Reward | Current State AND (Current) Action]

import numpy as np
import numba
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

#%%# Class Environment [Grid World | Finite Markov Decision Process | MDP (State) Transition Probabilities]

class GridWorld:
    
    def __init__(self, rows, cols):
        self.rows = rows # i ~ row
        self.cols = cols # j ~ col
    
    def states_set(self):
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)] # Default!
        self.state_space = dict(enumerate(self.states)) # (cols * i + j, (i, j))
        return self
    
    def state_forbidden_set(self, state_forbidden = None):
        _state_forbidden = [(1, 1)] # Default!
        if state_forbidden is None:
            state_forbidden = {state: False if state not in _state_forbidden else True for state in self.states}
        else:
            _mess = 'User Interaction!\n\t'
            mess = _mess + f'Forbidden State\n{state_forbidden}'
            print(mess)
        _mess = 'Oops! The user definition is invalid.\n\t'
        mess = _mess + "We must provide a 'dictionary'!"
        check = type(state_forbidden) is dict
        assert check, mess
        mess = _mess + 'We must provide a Boolean value for every state!'
        check_alp = all([key in self.states for key in state_forbidden.keys()])
        check_bet = all([type(value) is bool for value in state_forbidden.values()])
        check = check_alp and check_bet
        assert check, mess
        self.state_forbidden = state_forbidden
        return self
    
    def state_terminal_set(self, state_terminal = None):
        _state_terminal = [(0, 3), (1, 3)] # Default!
        if state_terminal is None:
            state_terminal = {state: False if state not in _state_terminal else True for state in self.states}
        else:
            _mess = 'User Interaction!\n\t'
            mess = _mess + f'Terminal State\n{state_terminal}'
            print(mess)
        _mess = 'Oops! The user definition is invalid.\n\t'
        mess = _mess + "We must provide a 'dictionary'!"
        check = type(state_terminal) is dict
        assert check, mess
        mess = _mess + 'A terminal state must not be a forbidden state!'
        check = all([not self.state_forbidden[state] for state in self.states if state_terminal[state]])
        assert check, mess
        mess = _mess + 'We must provide a Boolean value for every state!'
        check_alp = all([key in self.states for key in state_terminal.keys()])
        check_bet = all([type(value) is bool for value in state_terminal.values()])
        check = check_alp and check_bet
        assert check, mess
        self.state_terminal = state_terminal
        return self
    
    def state_terminal_special_set(self, state_terminal_special = None):
        _state_terminal_special = [(1, 3)] # Default!
        if state_terminal_special is None:
            state_terminal_special = {state: False if state not in _state_terminal_special else True for state in self.states}
        else:
            _mess = 'User Interaction!\n\t'
            mess = _mess + f'Special Terminal State\n{state_terminal_special}'
            print(mess)
        _mess = 'Oops! The user definition is invalid.\n\t'
        mess = _mess + "We must provide a 'dictionary'!"
        check = type(state_terminal_special) is dict
        assert check, mess
        mess = _mess + 'A special terminal state must be a terminal state!'
        check = all([self.state_terminal[state] for state in self.states if state_terminal_special[state]])
        assert check, mess
        mess = _mess + 'We must provide a Boolean value for every state!'
        check_alp = all([key in self.states for key in state_terminal_special.keys()])
        check_bet = all([type(value) is bool for value in state_terminal_special.values()])
        check = check_alp and check_bet
        assert check, mess
        self.state_terminal_special = state_terminal_special
        return self
    
    def state_start_set(self, state_start = None):
        _state_start = (self.rows - 1, 0) # Default! # (row, col)
        if state_start is None:
            state_start = _state_start
        else:
            _mess = 'User Interaction!\n\t'
            mess = _mess + f'Start State\n{state_start}'
            print(mess)
        _mess = 'Oops! The user definition is invalid.\n\t'
        mess = _mess + "We must provide a 'duple'!"
        check_alp = type(state_start) is tuple
        check_bet = len(state_start) == 2
        check = check_alp and check_bet
        assert check, mess
        mess = _mess + 'The start state must be an element of the state space!'
        check = state_start in self.states
        assert check, mess
        mess = _mess + 'The start state is neither a forbidden state nor a terminal state!'
        check_alp = not self.state_forbidden[state_start]
        check_bet = not self.state_terminal[state_start]
        check = check_alp and check_bet
        assert check, mess
        self.state_start = state_start
        self.i = state_start[0] # row
        self.j = state_start[1] # col
        return self
    
    def state_get_current(self):
        current_state = (self.i, self.j)
        return current_state
    
    def state_get_next(self, current_state, action):
        i, j = current_state
        if action in self.actions[current_state].keys():
            i += self.action_space_row.get(action, 0)
            j += self.action_space_col.get(action, 0)
        next_state = (i, j)
        return next_state
    
    def state_reset(self):
        self.i = self.state_start[0] # row
        self.j = self.state_start[1] # col
        return self
    
    def actions_set(self, actions = None):
        action_space = {'U': -1, 'D': 1, 'L': -1, 'R': 1} # Default! # {'U': 'Up', 'D': 'Down', 'L': 'Left', 'R': 'Right'}
        self.action_space = dict(enumerate(action_space.items()))
        self.action_space_row = {key: value for key, value in action_space.items() if key in ['U', 'D']}
        self.action_space_col = {key: value for key, value in action_space.items() if key in ['L', 'R']}
        if actions is None:
            actions = dict()
            rows = range(self.rows)
            cols = range(self.cols)
            for state in self.states:
                i, j = state
                skip = []
                for key, value in action_space.items():
                    if key in self.action_space_row.keys():
                        row = i + value
                        if row not in rows or self.state_forbidden[(row, j)]:
                            skip.append(key)
                    else: # key in self.action_space_col.keys()
                        col = j + value
                        if col not in cols or self.state_forbidden[(i, col)]:
                            skip.append(key)
                if self.state_forbidden[state] or self.state_terminal[state]:
                    _action_space = dict()
                else:
                    _action_space = {key: value for key, value in action_space.items() if key not in skip}
                actions.update({state: _action_space})
        else:
            _mess = 'User Interaction!\n\t'
            mess = _mess + f'Actions\n{actions}'
            print(mess)
        _mess = 'Oops! The user definition is invalid.\n\t'
        mess = _mess + "We must provide a 'dictionary'!"
        check = type(actions) is dict
        assert check, mess
        mess = _mess + 'We must provide a dictionary of the actions for every state!'
        check_alp = all([key in self.states for key in actions.keys()])
        check_bet = all([type(value) is dict for value in actions.values()])
        check = check_alp and check_bet
        assert check, mess
        mess = _mess + 'Every action must belong to the action space!'
        for temp in actions.values():
            check_alp = all([key in action_space.keys() for key in temp.keys()])
            check_bet = all([value in set(action_space.values()) for value in temp.values()])
            check = check_alp and check_bet
            assert check, mess
        self.actions = actions
        return self
    
    def rewards_set(self, rewards = None):
        reward_space = {'Nonterminal_State': 0, 'Forbidden_State': None, 'Terminal_State': 1, 'Special_Terminal_State': -1} # Default!
        self.reward_space = dict(enumerate(reward_space.items()))
        if rewards is None:
            rewards = {state: None for state in self.states}
            rewards.update({state: reward_space['Nonterminal_State'] for state in self.states if not self.state_forbidden[state] and not self.state_terminal[state]})
            rewards.update({state: reward_space['Terminal_State'] for state in self.states if self.state_terminal[state] and not self.state_terminal_special[state]})
            rewards.update({state: reward_space['Special_Terminal_State'] for state in self.states if self.state_terminal_special[state]})
        else:
            _mess = 'User Interaction!\n\t'
            mess = _mess + f'Rewards\n{rewards}'
            print(mess)
        _mess = 'Oops! The user definition is invalid.\n\t'
        mess = _mess + "We must provide a 'dictionary'!"
        check = type(rewards) is dict
        assert check, mess
        mess = _mess + 'We must provide a reward for every state!'
        check = all([key in self.states for key in rewards.keys()])
        assert check, mess
        self.rewards = rewards
        return self
    
    def game_over(self):
        state = self.state_get_current()
        over = self.state_terminal[state]
        return over
    
    def assemble(self, verbose = False):
        self.states_set()
        self.state_forbidden_set()
        self.state_terminal_set()
        self.state_terminal_special_set()
        self.state_start_set()
        self.actions_set()
        self.rewards_set()
        if verbose:
            actions = list(self.action_space_row.keys()) + list(self.action_space_col.keys())
            stretch = 7 # print(stretch % 2 == 1) # print(stretch >= 7)
            mess = '\n' + '%' * stretch + ' Grid World: Descript! ' + '%' * stretch + '\n'
            print(mess)
            for state in self.states:
                mess = state
                print(mess)
                for action in actions:
                    mess = '\t\t' + action
                    print(mess, end = '')
                    i, j = state
                    if action in self.actions[state].keys():
                        i += self.action_space_row.get(action, 0)
                        j += self.action_space_col.get(action, 0)
                        mess = '\t' + str((i, j))
                        print(mess)
                    else:
                        mess = '\t' + '-' * 8
                        print(mess)
        return self
    
    def dynamics(self): # (MDP | Environment) Dynamics (Function) # p(s', r | s, a)
        transition_probabilities = dict() # (State) Transition Probabilities # p(s' | s, a)
        rewards = dict() # (Expected) Rewards # r(s, a, s')
        for i in range(self.rows):
            for j in range(self.cols):
                current_state = (i, j)
                for action in self.actions[current_state].keys():
                    next_state = self.state_get_next(current_state, action)
                    key = (current_state, action, next_state) # (s, a, s') # Triple!
                    value = 1
                    transition_probabilities[key] = value
                    value = self.rewards[next_state]
                    rewards[key] = value
        dynamicist = (transition_probabilities, rewards)
        return dynamicist
    
    def _drawer(self, stretch, tit): # _sketcher
        draw = dict()
        _draw = ['\n', '%'*stretch, ' ', tit, ' ', '%'*stretch, '\n']
        draw.update({'tit': ''.join(_draw)})
        _draw = [' '*int(stretch/2)+f'{j}'+' '*int(stretch/2) for j in range(self.cols)]
        draw.update({'col_indices': ' '+''.join(_draw)})
        draw.update({'bound': ' '+'-'*stretch*self.cols})
        draw.update({'top': '\n'+' '+('|'+' '*(stretch-2)+'|')*self.cols})
        draw.update({'bot': '\n'+' '+('|'+' '*(int(stretch/2)-1)+' '+' '*(int(stretch/2)-1)+'|')*self.cols})
        return draw
    
    def draw_actions(self):
        stretch = 11 # print(stretch % 2 == 1) # print(stretch >= 7)
        tit = 'Grid World: Actions!'
        draw = self._drawer(stretch, tit)
        print(draw['tit'], draw['col_indices'], sep = '\n')
        A = '·' # Default!
        Z = ' ' # Default!
        for i in range(self.rows):
            print(draw['bound'], draw['top'], sep = '')
            draw.update({'row_index': str(i)})
            print(' ', end = '')
            for j in range(self.cols):
                state = (i, j)
                test = self.state_forbidden[state] or self.state_terminal[state]
                U = 'U' if 'U' in self.actions[state].keys() else A
                mess = U if not test else Z
                orbit = int((stretch-len(mess)-2)/2)
                _draw = ['|', ' '*orbit, mess, ' '*orbit, '|']
                draw.update({'mid': ''.join(_draw)})
                print(draw['mid'], end = '')
            print('')
            for j in range(self.cols):
                state = (i, j)
                test = self.state_forbidden[state] or self.state_terminal[state]
                L = 'L' if 'L' in self.actions[state].keys() else A
                C = A
                R = 'R' if 'R' in self.actions[state].keys() else A
                mess = L+' '+C+' '+R if not test else Z
                orbit = int((stretch-len(mess)-2)/2)
                _draw = ['|', ' '*orbit, mess, ' '*orbit, '|']
                draw.update({'mid': ''.join(_draw)})
                if j == 0:
                    print(draw['row_index']+draw['mid'], end = '')
                else:
                    print(draw['mid'], end = '')
            print('')
            print(' ', end = '')
            for j in range(self.cols):
                state = (i, j)
                test = self.state_forbidden[state] or self.state_terminal[state]
                D = 'D' if 'D' in self.actions[state].keys() else A
                mess = D if not test else Z
                orbit = int((stretch-len(mess)-2)/2)
                _draw = ['|', ' '*orbit, mess, ' '*orbit, '|']
                draw.update({'mid': ''.join(_draw)})
                print(draw['mid'], end = '')
            print(draw['bot'])
        print(draw['bound'])
        return None
    
    def draw_value_function(self, V):
        stretch = 11 # print(stretch % 2 == 1) # print(stretch >= 7)
        tit = 'Grid World: Value Function!'
        draw = self._drawer(stretch, tit)
        print(draw['tit'], draw['col_indices'], sep = '\n')
        cushion = 4 # Default!
        for i in range(self.rows):
            print(draw['bound'], draw['top']*2, sep = '')
            draw.update({'row_index': str(i)})
            for j in range(self.cols):
                state = (i, j)
                v = V[state]
                if self.state_forbidden[state]:
                    mess = ' '
                elif self.state_terminal[state]:
                    mess = '0'
                else:
                    _mess = '+' if v > 0 else ''
                    mess = _mess+str(np.round(float(v), cushion))
                    dit = stretch-cushion-len(mess)
                    if v != 0:
                        if dit > 0:
                            mess += '0'*dit
                        elif dit < 0:
                            mess = mess[0:dit] if abs(v) < 1E4 else '{:+.2E}'.format(v)
                orbit = int((stretch-len(mess)-2)/2)
                _draw = ['|', ' '*orbit, mess, ' '*orbit, '|']
                draw.update({'mid': ''.join(_draw)})
                if j == 0:
                    print(draw['row_index']+draw['mid'], end = '')
                else:
                    print(draw['mid'], end = '')
            print(draw['bot']*2)
        print(draw['bound'])
        return None
    
    def draw_policy(self, pi):
        stretch = 11 # print(stretch % 2 == 1) # print(stretch >= 7)
        tit = 'Grid World: Policy!'
        draw = self._drawer(stretch, tit)
        print(draw['tit'], draw['col_indices'], sep = '\n')
        for i in range(self.rows):
            print(draw['bound'], draw['top']*2, sep = '')
            draw.update({'row_index': str(i)})
            for j in range(self.cols):
                state = (i, j)
                if self.state_forbidden[state] or self.state_terminal[state]:
                    mess = ' '
                else:
                    mess = pi[state]
                _draw = ['|', ' '*(int(stretch/2)-1), mess, ' '*(int(stretch/2)-1), '|']
                draw.update({'mid': ''.join(_draw)})
                if j == 0:
                    print(draw['row_index']+draw['mid'], end = '')
                else:
                    print(draw['mid'], end = '')
            print(draw['bot']*2)
        print(draw['bound'])
        return None
