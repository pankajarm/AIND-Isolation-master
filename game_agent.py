"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # let's begin.....

    # return defaults states
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    # let's get our and opponent moves
    our_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # main meat here, a future place for awesome deep learning powers....
    return float(our_moves - (2.0 * opp_moves))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # let's begin.....

    # return defaults states
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    # let's get our and opponent moves
    our_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # main meat here, a future place for awesome deep learning powers....
    return float(our_moves / (opp_moves + 0.0001))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # let's begin.....

    # return defaults states
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    # let's get our and opponent moves
    our_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # main meat here, a future place for awesome deep learning powers....
    return float(our_moves - opp_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # let's begin...

        # get the legal moves
        legal_moves = game.get_legal_moves()
        # Initialize the default best move
        best_move = (-1,-1)
        # Initialize the minmax score variable
        minmax_score = float("-inf")

        # loop through legal moves to determine the best moves
        for move in legal_moves:
            new_score = self.min_value(game.forecast_move(move), depth-1)
            # update minmax_score if the current move is better
            if new_score > minmax_score:
                minmax_score = new_score
                best_move = move
        # return the best move
        return best_move
    # defined max_value function as described in AIMA book
    def max_value(self, game, depth):

        # check for timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # get the legal moves
        legal_moves = game.get_legal_moves()
        # terminal tests
        if not legal_moves:
            return game.utility(self)
        # if terminal state is depth limit then return current utility score
        if depth == 0:
            return self.score(game, self)
        # get the min_score variable
        max_score = float("-inf")
        # loop through the legal moves to determine min score
        for move in legal_moves:
            value = self.min_value(game.forecast_move(move), depth-1)
            # update best score
            if value > max_score:
                max_score = value

        return max_score

    # defined min_value function as described in AIMA book
    def min_value(self, game, depth):

        # check for timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # get the legal moves
        legal_moves = game.get_legal_moves()
        # terminal tests
        if not legal_moves:
            return game.utility(self)
        # if terminal state is depth limit then return current utility score
        if depth == 0:
            return self.score(game, self)
        # get the min_score variable
        min_score = float("inf")
        # loop through the legal moves to determine min score
        for move in legal_moves:
            value = self.max_value(game.forecast_move(move), depth-1)
            # update best score
            if value < min_score:
                min_score = value
        return min_score



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # let's begin....

        # get the legal moves
        legal_moves = game.get_legal_moves()

        # check if there are legal moves
        if not legal_moves:
            return (-1,-1)

        # set the best move with a random legal move
        best_move = random.choice(legal_moves)

        # let's start with depth = 1
        depth = 1
        # Let's use iterative deepening in given time
        try:
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            pass

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # let's begin...
        # get the legal moves
        legal_moves = game.get_legal_moves()
        # Initialize default alphabeta_value
        alphabeta_score = float("-inf")
        best_move = (-1,-1)
        # terminal test
        if not legal_moves:
            return best_move

        # terminal test for depth: return the first move
        if depth == 0:
            return  legal_moves[0]

        new_score=0
        for move in legal_moves:
            # get minimium score for each forecasted move
            new_score = self.min_value(game.forecast_move(move), depth-1, alpha, beta)
            # assign best move is current if alphabeta_value is greater then return new move
            if new_score > alphabeta_score:
                alphabeta_score = new_score
                best_move = move
            # if current alphabeta_score is greater then beta then return default move
            if alphabeta_score >= beta:
                return best_move
            # choose alpha between currebnt and the max of new score
            alpha = max(alpha, new_score)
        return best_move

    # defined max_value function as described in AIMA book
    def max_value(self, game, depth, alpha, beta):
        # check for timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # get the legal moves
        legal_moves = game.get_legal_moves()
        # terminal test no legal move left
        if not legal_moves:
            return game.utility(self)
        # check for terminal state
        if depth == 0:
            return self.score(game,self)
        # if no timeout and no terminal then let's move forward
        # Initialize max_score variable
        max_score = float("-inf")
        # loop through all the moves in game get_legal_moves
        for move in legal_moves:
            # update max_score from local min_value method
            max_score = max(max_score, self.min_value(game.forecast_move(move), depth-1, alpha, beta))
            # check if max_score is still greater then current beta score
            if max_score >= beta:
                return max_score
            # otherwise update alpha with max_score
            alpha = max(alpha, max_score)
        # and return max_score
        return max_score

    # defined min_value function as described in AIMA book
    def min_value(self, game, depth, alpha, beta):

        # check for timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # get the legal moves
        legal_moves = game.get_legal_moves()
        # terminal test no legal move left
        if not legal_moves:
            return game.utility(self)
        # check for terminal state
        if depth == 0:
            return self.score(game, self)

        # if no timeout and no terminal then let's move forward
        # Initialize min_score variable
        min_score = float("inf")
        # loop through all the moves in game get_legal_moves
        for move in legal_moves:
            # update min_score local max_value method
            min_score = min(min_score, self.max_value(game.forecast_move(move), depth-1, alpha, beta))
            # check if min_score is still less then current alpha score
            if min_score <= alpha:
                return min_score
            # otherwise update beta score with min_score
            beta = min(beta, min_score)
        # and return min_score
        return min_score
