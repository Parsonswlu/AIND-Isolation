"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    custom_score_choice = 3

    # Option #1 - difference in own and opp moves plus overlapped moves
    if custom_score_choice == 1:
        own_moves = game.get_legal_moves(player)
        opp_moves = game.get_legal_moves(game.get_opponent(player))
        overlap = set(own_moves).intersection(opp_moves)
        return float(len(own_moves) - len(opp_moves) + len(list(overlap)))

    # Option #2 - check diamond area of immediate vicinity
    elif custom_score_choice == 2:
        own_moves = game.get_legal_moves(player)
        opp_moves = game.get_legal_moves(game.get_opponent(player))
        diamond_moves = [(2,0),(-2,0),(0,2),(0,-2),(1,1),(-1,1),(1,-1),(-1,-1)]
        own_loc = game.get_player_location(player)
        opp_loc = game.get_player_location(game.get_opponent(player))

        for x in diamond_moves:
            if game.move_is_legal(tuple(map(sum,zip(x,own_loc)))):
                own_moves+=tuple(map(sum,zip(x,own_loc)))
            if game.move_is_legal(tuple(map(sum,zip(x,opp_loc)))):
                opp_moves+=tuple(map(sum,zip(x,opp_loc)))

        return float(len(own_moves) - 2*len(opp_moves))

    # Option #3 - diamond moves + give higher scores to moves that get more aggressive as the game goes on
    elif custom_score_choice == 3:
        own_moves = game.get_legal_moves(player)
        opp_moves = game.get_legal_moves(game.get_opponent(player))
        diamond_moves = [(2,0),(-2,0),(0,2),(0,-2),(1,1),(-1,1),(1,-1),(-1,-1)]
        own_loc = game.get_player_location(player)
        opp_loc = game.get_player_location(game.get_opponent(player))

        for x in diamond_moves:
            if game.move_is_legal(tuple(map(sum,zip(x,own_loc)))):
                own_moves+=tuple(map(sum,zip(x,own_loc)))
            if game.move_is_legal(tuple(map(sum,zip(x,opp_loc)))):
                opp_moves+=tuple(map(sum,zip(x,opp_loc)))

        game_progress = game.move_count / (game.width*game.height)

        return float(len(own_moves) - 3*game_progress*len(opp_moves))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!
        action = (-1,-1)

        if not legal_moves:
            return action   
        
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            depth = 1 if self.iterative else self.search_depth
            while True: 
                if self.method == 'minimax':
                    v, action = self.minimax(game,depth)
                elif self.method == 'alphabeta':
                    v, action = self.alphabeta(game,depth)
                if not self.iterative:
                    return action
                depth+=1

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return action

        # Return the best move from the last completed search iteration
        return action

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        ## The code below is based partially on pseudocode from Section 5.3 of the textbook 
        ## AIMA: 3rd edition by Stuart J Russell and Peter Norvig
        legal_moves = game.get_legal_moves()
        best_score = float("-inf") if maximizing_player else float("inf")
        best_action = (-1,-1)

        if not legal_moves or depth == 0:
            return self.score(game,self), best_action

        for action in legal_moves:
            # Recursively check values in child nodes
            v, _ = self.minimax(game.forecast_move(action),depth-1,maximizing_player = not maximizing_player)
            if (v > best_score and maximizing_player) or (v < best_score and not maximizing_player):
                best_score, best_action = v, action
        return best_score, best_action

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        ## The code below is based partially on pseudocode from Section 5.3 of the textbook 
        ## AIMA: 3rd edition by Stuart J Russell and Peter Norvig
        legal_moves = game.get_legal_moves()
        best_score = float("-inf") if maximizing_player else float("inf")
        best_action = (-1,-1)

        ## Cutoff Test
        if not legal_moves or depth == 0:
            return self.score(game,self), best_action

        for action in legal_moves:
            # Recursively check values in child nodes
            v, _ = self.alphabeta(game.forecast_move(action),depth-1,alpha,beta,maximizing_player = not maximizing_player)
            if maximizing_player:
                # Prune if possible
                if v >= beta:
                    return v, action
                # Otherwise update score if better than previous
                if v > best_score:
                    best_score, best_action = v, action
                alpha = max(alpha,v)
            elif not maximizing_player:
                # Prune if possible
                if v <= alpha:
                    return v, action
                # Otherwise update score if better than previouss
                if v < best_score:
                    best_score, best_action = v, action
                beta = min(beta,v)                
        return best_score, best_action
