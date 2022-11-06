__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "July 2022"

# Import the random number generation library
import random
import helper
import numpy as np

def miniMax(maximising, state, depth):
    if helper.terminal(state) or depth == 0:
        return helper.evaluate(state)

    scores = []
    if maximising:
        possible_moves = helper.remove_symmetries(helper.maxs_possible_moves(state))
    else:
        possible_moves = helper.remove_symmetries(helper.mins_possible_moves(state))

    for move in possible_moves:
        scores.append(miniMax(not maximising, move, depth - 1))

    if maximising:
        index = np.argmax(scores)
    else:
        index = np.argmin(scores)

    return helper.evaluate(possible_moves[index])



class TicTacToeAgent():
    """
           A class that encapsulates the code dictating the
           behaviour of the TicTacToe playing agent

           Methods
           -------
           AgentFunction(percepts)
               Returns the move made by the agent given state of the game in percepts
           """

    def __init__(self, h):
        """Initialises the agent

        :param h: Handle to the figures showing state of the board -- only used
                  for human_agent.py to enable selecting next move by clicking
                  on the matplotlib figure.
        """
        pass



    def AgentFunction(self, percepts):
        """The agent function of the TicTacToe agent -- returns action
         relating the row and column of where to make the next move

        :param percepts: the state of the board a list of rows, each
        containing a value of three columns, where 0 identifies the empty
        suare, 1 is a square with this agent's mark and -1 is a square with
        opponent's mark
        :return: tuple (r,c) where r is the row and c is the column index
                 where this agent wants to place its mark
        """
        new_states = helper.remove_symmetries(helper.maxs_possible_moves(percepts))
        scores = []
        for state in new_states:
            score = miniMax(False, state, 1000)
            scores.append(score)

        index = np.argmax(scores)

        r, c = helper.state_change_to_action(percepts, new_states[index])
        return r, c
