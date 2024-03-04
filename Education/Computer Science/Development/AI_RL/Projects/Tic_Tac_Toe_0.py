#############################
######## Tic Tac Toe ########
#############################

#%%# Catalyzer

import numpy as np
# import numba
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

board_length = 3

#%%# Class Agent [Epsilon-Greedy | Action-Value Methods]

class Agent:
    
    def __init__(self, epsilon = 0.1, alpha = 0.5, board_length = board_length):
        self.epsilon = epsilon
        self.alpha = alpha # Step-Size Parameter # It influences the rate of learning!
        self.board_length = board_length
        self.state_history = []
        self.verbose = False
    
    def V_set(self, V):
        self.V = V # Value Function
        return self
    
    def symbol_set(self, symbol):
        self.symbol = symbol # [-1, 1]
        return self
    
    def verbose_set(self, verbose):
        self.verbose = verbose
        return self
    
    def state_history_update(self, state):
        self.state_history.append(state)
        return self
    
    def state_history_reset(self):
        self.state_history = []
        return self
    
    def random_sample_draw(self, size = 1, seed = None): # Random Sample (Generate)
        self.random_sample = np.random.default_rng(seed = seed).random(size = size) # Continuous Uniform Random Vector
        return self
    
    def action_select(self, environment): # Action := Move
        # Epsilon-Greedy Strategy
        self.random_sample_draw()
        if self.random_sample < self.epsilon:
            # Uniform Random Action Selection
            if self.verbose:
                print('Epsilon-Greedy Strategy\n\tUniform Random Action Selection! Exploratory Move!')
            possible_moves = []
            for i in range(self.board_length):
                for j in range(self.board_length):
                    if environment.board_position_empty(i, j):
                        move = (i, j)
                        possible_moves.append(move)
            _next_move = np.random.default_rng(seed = None).choice(a = len(possible_moves), size = 1).item() # Discrete Uniform Random Vector
            next_move = possible_moves[_next_move]
        else:
            # Greedy Action Selection
            if self.verbose:
                print('Epsilon-Greedy Strategy\n\tGreedy Action Selection! Greedy Move!')
                print('To select our moves we examine the states that would result from each of our possible moves (one for each blank space on the board) and look up their current values in the table.')
            state_value = dict()
            best_state_value = -1
            for i in range(self.board_length):
                for j in range(self.board_length):
                    if environment.board_position_empty(i, j):
                        environment.board[i, j] = self.symbol
                        state = environment.state_get()
                        state_value.update({(i, j): self.V[state]})
                        environment.board[i, j] = 0
                        if self.V[state] > best_state_value:
                            best_state_value = self.V[state]
                            best_state = state
                            next_move = (i, j)
            if self.verbose:
                print(f'Move {next_move}\n\tState {best_state}\n\t\tState Value {best_state_value}')
                # Draw Board
                stretch = 11 # print(stretch % 2 == 1) # print(stretch >= 7)
                draw = dict()
                _draw = ['\n', '%'*stretch, ' Tic-Tac-Toe Board ', '%'*stretch, '\n']
                draw.update({'tit': ''.join(_draw)})
                _draw = [' '*int(stretch/2)+f'{j}'+' '*int(stretch/2) for j in range(self.board_length)]
                draw.update({'col_indices': ' '+''.join(_draw)})
                draw.update({'bound': ' '+'-'*stretch*self.board_length})
                draw.update({'top': '\n'+' '+('|'+' '*(stretch-2)+'|')*self.board_length})
                draw.update({'bot': '\n'+' '+('|'+' '*(int(stretch/2)-1)+' '+' '*(int(stretch/2)-1)+'|')*self.board_length})
                print(draw['tit'], draw['col_indices'], sep = '\n')
                for i in range(self.board_length):
                    print(draw['bound'], draw['top']*2, sep = '')
                    draw.update({'row_index': str(i)})
                    for j in range(self.board_length):
                        try:
                            mess = str(np.round(state_value[(i, j)], 3)) # mess = f'{i}'+' '*3+f'{j}'
                            dit = 5-len(mess)
                            if dit > 0:
                                mess += '0'*dit
                        except KeyError:
                            mess = ' '*5
                        _draw = ['|', ' '*(int(stretch/2)-3), mess, ' '*(int(stretch/2)-3), '|']
                        draw.update({'mid': ''.join(_draw)})
                        if j == 0:
                            print(draw['row_index']+draw['mid'], end = '')
                        else:
                            print(draw['mid'], end = '')
                    print(draw['bot']*2)
                print(draw['bound'])
        environment.board[next_move[0], next_move[1]] = self.symbol
        return self
    
    def V_update(self, environment):
        step_size = self.alpha
        reward = environment.reward_get(self.symbol)
        target = reward
        for previous_state in reversed(self.state_history):
            old_estimate = self.V[previous_state]
            new_estimate = old_estimate + step_size * (target - old_estimate) # Update Rule # Temporal-Difference Learning Methods
            self.V[previous_state] = new_estimate
            target = new_estimate
        self.state_history_reset()
        return self

#%%# Class Environment

class Environment:
    
    def __init__(self, board_length = board_length):
        self.board_length = board_length
        self.board = np.zeros((self.board_length, self.board_length))
        self.player_alp = -1 # X
        self.player_bet = 1 # Q
        self._player_alp = 'X' # -1
        self._player_bet = 'Q' # 1
        self.winner = None
        self.game_end = False
        self.state_cardinality = np.power(self.board_length, np.power(self.board_length, 2))
    
    def board_position_empty(self, i, j):
        position_empty = self.board[i, j] == 0 # State := Board Position
        return position_empty
    
    def state_get(self): # print(self.board_length == 3)
        player_weights = np.arange(self.board_length) # [0, ..., self.board_length-1]
        player_ids = [0, self.player_alp, self.player_bet] # [player_none, -1, 1]
        players = []
        for w in player_weights:
            board_positions = np.argwhere(self.board.ravel() == player_ids[w]).ravel()
            players.append(np.sum(w*np.power(self.board_length, board_positions)))
        state = np.sum(players)
        return state
    
    def reward_get(self, symbol):
        if not self.game_over():
            reward = 0
            return reward
        if self.winner == symbol:
            reward = 1
        else:
            reward = 0
        return reward
    
    def game_over(self, recalculate = False):
        # Recalculation
        if not recalculate and self.game_end:
            over = True
            return over
        players = [self.player_alp, self.player_bet]
        # Check Board Rows
        rows = np.sum(self.board, 0)
        # Check Board Columns
        cols = np.sum(self.board, 1)
        # Check Board Diagonals
        diagonals = [np.trace(self.board), np.trace(np.flip(self.board, 1))]
        # Check Whole Board
        whole = np.concatenate((rows, cols, diagonals))
        for player in players:
            if player*self.board_length in whole:
                self.winner = player
                self.game_end = True
                over = True
                return over
        # Draw/Tie?
        if np.all(self.board != 0):
            self.winner = None
            self.game_end = True
            over = True
            return over
        # The game is not over!
        self.winner = None
        over = False
        return over
    
    def board_draw(self):
        stretch = 11 # print(stretch % 2 == 1) # print(stretch >= 7)
        draw = dict()
        _draw = ['\n', '%'*stretch, ' Tic-Tac-Toe Board ', '%'*stretch, '\n']
        draw.update({'tit': ''.join(_draw)})
        _draw = [' '*int(stretch/2)+f'{j}'+' '*int(stretch/2) for j in range(self.board_length)]
        draw.update({'col_indices': ' '+''.join(_draw)})
        draw.update({'bound': ' '+'-'*stretch*self.board_length})
        draw.update({'top': '\n'+' '+('|'+' '*(stretch-2)+'|')*self.board_length})
        draw.update({'bot': '\n'+' '+('|'+' '*(int(stretch/2)-1)+' '+' '*(int(stretch/2)-1)+'|')*self.board_length})
        print(draw['tit'], draw['col_indices'], sep = '\n')
        for i in range(self.board_length):
            print(draw['bound'], draw['top']*2, sep = '')
            draw.update({'row_index': str(i)})
            for j in range(self.board_length):
                if self.board[i, j] == self.player_alp:
                    mess = self._player_alp
                elif self.board[i, j] == self.player_bet:
                    mess = self._player_bet
                else:
                    mess = ' ' # _player_none
                _draw = ['|', ' '*(int(stretch/2)-1), mess, ' '*(int(stretch/2)-1), '|']
                draw.update({'mid': ''.join(_draw)})
                if j == 0:
                    print(draw['row_index']+draw['mid'], end = '')
                else:
                    print(draw['mid'], end = '')
            print(draw['bot']*2)
        print(draw['bound'])
        return None

#%%# Class Human

class Human:
    
    def __init__(self):
        pass
    
    def symbol_set(self, symbol):
        self.symbol = symbol # [-1, 1]
        return self
    
    def state_history_update(self, state):
        pass
    
    def action_select(self, environment):
        roc = np.arange(environment.board_length)
        flag = False
        while not flag:
            row = int(input('%%%%%%%% Row? '))
            col = int(input('%%%%%%%% Col? '))
            move = (row, col)
            test = row in roc and col in roc and environment.board_position_empty(row, col)
            if test:
                print(f'\nValid Move!\t(Row, Col) = {move}\n')
                next_move = move
                flag = True
            else:
                print(f'\nInvalid Move!\t(Row, Col) = {move}')
        environment.board[next_move[0], next_move[1]] = self.symbol
        return self
    
    def V_update(self, environment):
        pass

#%%# Function Game Statuses

def _translate(b = 3, c = 3**2, d = 0):
    # descript = {'b': 'base', 'c': {'board positions', 'state cardinality'}, 'd': 'decimal'}
    if d > np.power(b, c) - 1:
        raise ValueError(f"The max value of 'd' must be '{np.power(b, c) - 1}'!")
    t = np.zeros(c) # ternary
    if d == 0:
        t[0] = 0
    else:
        i = 0 # index
        while d != 0:
            q = d // b # quotient
            r = d % b # remainder
            # print(q, r)
            d = q
            t[i] = r
            i += 1
    return t

def _game_impossible(environment, board):
    impossible = False
    player_moves = np.unique(ar = board, return_counts = True)
    player_alp = np.where(player_moves[0] == environment.player_alp, True, False)
    player_bet = np.where(player_moves[0] == environment.player_bet, True, False)
    player_alp_moves = player_moves[1][player_alp] if np.any(player_alp) else 0
    player_bet_moves = player_moves[1][player_bet] if np.any(player_bet) else 0
    if np.abs(player_alp_moves - player_bet_moves) > 1:
        impossible = True
    return impossible

def _game_impossible_correction(board, winner):
    impossible_correction = False
    player_moves = np.unique(ar = board, return_counts = True)
    player_winner = np.where(player_moves[0] == winner, True, False)
    player_loser = np.where(player_moves[0] == -1*winner, True, False)
    player_winner_moves = player_moves[1][player_winner]
    player_loser_moves = player_moves[1][player_loser]
    if player_winner_moves - player_loser_moves < 0:
        impossible_correction = True
    return impossible_correction

def game_statuses(environment, show = False):
    states = range(np.power(environment.board_length, environment.board_length**2))
    statuses = np.zeros(shape = (len(states), 3), dtype = np.int8) # [winner, game_end, game_impossible] # [{-1, 0, 1}, {0, 1}, {0, 1}]
    for state in states:
        if show: print(state)
        t = _translate(d = state)
        board = np.where(t == 1, -1, np.where(t == 2, 1, t)).reshape((environment.board_length, environment.board_length))
        if show: print(board)
        impossible = _game_impossible(environment, board)
        if impossible:
            winner = 0 # None
            game_end = 1 # True
            game_impossible = 1 # True
            statuses[state] = [winner, game_end, game_impossible]
            if show: print(statuses[state])
            continue
        game_moves = np.count_nonzero(board)
        if game_moves >= 5:
            rows = np.sum(board, 0)
            cols = np.sum(board, 1)
            diagonals = [np.trace(board), np.trace(np.flip(board, 1))]
            whole = np.concatenate((rows, cols, diagonals))
            if show: print(whole)
            test = np.count_nonzero(np.abs(whole) == environment.board_length)
            if test != 1:
                winner = 0 # None
                if test < 1:
                    game_end = 1 if game_moves == 9 else 0 # [True, False]
                    game_impossible = 0 # False
                else: # test > 1
                    game_end = 1 # True
                    game_impossible = 1 # True
            else: # test == 1
                winner = int(np.sign(whole[np.argmax(np.abs(whole))])) # {-1, 1}
                game_end = 1 # True
                game_impossible = 0 # False
            if winner != 0:
                impossible_correction = _game_impossible_correction(board, winner)
                if impossible_correction:
                    winner = 0 # None
                    game_end = 1 # True
                    game_impossible = 1 # True
        else: # game_moves < 5
            winner = 0 # None
            game_end = 0 # False
            game_impossible = 0 # False
        statuses[state] = [winner, game_end, game_impossible]
        if show: print(statuses[state])
    return statuses

#%%# Function V Instate

def V_instate(statuses, player):
    # descript = {'win': 1, {'lose', 'draw', 'impossible'}: 0, 'other': 0.5}
    V = np.zeros(statuses.shape[0])
    win = np.argwhere(statuses[..., 0] == player).ravel()
    lose = np.argwhere(statuses[..., 0] == -1*player).ravel()
    _draw = np.logical_and(np.logical_and(statuses[..., 0] == 0, statuses[..., 1] == 1), statuses[..., 2] == 0)
    draw = np.argwhere(_draw).ravel()
    impossible = np.argwhere(statuses[..., 2] == 1).ravel()
    _other = np.logical_and(np.logical_and(statuses[..., 0] == 0, statuses[..., 1] == 0), statuses[..., 2] == 0)
    other = np.argwhere(_other).ravel()
    check_alp = len(win) + len(lose) + len(draw) + len(impossible) + len(other) == statuses.shape[0]
    check_bet = np.all(np.equal(np.sort(np.concatenate((win, lose, draw, impossible, other))), np.arange(statuses.shape[0])))
    check = check_alp and check_bet
    assert check, 'Oops! Our logic is incorrect. We must fix our data filters!'
    V[win] = 1
    V[lose] = 0
    V[draw] = 0
    V[impossible] = 0
    V[other] = 0.5
    return V

#%%# Function Game Play [Agent | Human]

def game_play(player_alp, player_bet, environment, draw_board = 0):
    # descript = {'draw_board': {0, -1, 1}}
    players = (player_alp, player_bet)
    # Play!
    flag = False
    while not flag:
        # player_turns = {'Player 1': player_alp, 'Player 2': player_bet}
        for player in players:
            # Draw Board
            if draw_board:
                if draw_board == environment.player_alp and player == player_alp:
                    environment.board_draw()
                if draw_board == environment.player_bet and player == player_bet:
                    environment.board_draw()
            # Select Action
            player.action_select(environment)
            # Update State History
            state = environment.state_get()
            player_alp.state_history_update(state)
            player_bet.state_history_update(state)
            flag = environment.game_over()
            if flag:
                break
    # Draw Board
    if draw_board:
        environment.board_draw()
    # Update V
    player_alp.V_update(environment)
    player_bet.V_update(environment)
    return None

#%%# Function Agent Train [Agent AGAINST Agent]

def agent_train(T = 10000, tau = 250):
    # T := Final Time Step + 1 := Time Steps Number # tau := Time Interval
    player_alp = Agent()
    player_bet = Agent()
    environment = Environment()
    statuses = game_statuses(environment)
    V_alp = V_instate(statuses, environment.player_alp)
    V_bet = V_instate(statuses, environment.player_bet)
    player_alp.V_set(V_alp)
    player_bet.V_set(V_bet)
    player_alp.symbol_set(environment.player_alp)
    player_bet.symbol_set(environment.player_bet)
    for t in range(T):
        if t % tau == 0:
            print(f'Train/Time Step {t}')
        game_play(player_alp, player_bet, environment)
        environment = Environment()
    agents = (player_alp, player_bet)
    return agents

#%%# Function Game Play Human [Agent AGAINST Human]

def game_play_human(agents):
    # player_turns = {'Player 1': agent, 'Player 2': human}
    environment = Environment()
    agent = agents[0] # player_alp
    agent.verbose_set(True)
    human = Human() # player_bet
    human.symbol_set(environment.player_bet)
    draw_board = 1 # {0: None, -1: player_alp, 1: player_bet}
    mess = {None: 'Tie', -1: 'Champ! Android!', 1: 'Champion! Human!'}
    flag = False
    while not flag:
        game_play(agent, human, environment, draw_board)
        print(f'\n%%%%%%%% {mess[environment.winner]} %%%%%%%%')
        persevere = input('%%%%%%%% Try again? ( [Y]es | [N]o ) ')
        test = persevere in ['Y', 'y', 'N', 'n']
        if test:
            if persevere in ['Y', 'y']:
                print("\nLet's play!")
                environment = Environment()
            else:
                print('\nThank you!')
                flag = True
        else:
            print("You must choose 'Y' or 'N'!")
    return None

#%%# Main Zero

if __name__ == '__main__':
    
    agents = agent_train()

#%%# Main One

if __name__ == '__main__':
    
    game_play_human(agents)
