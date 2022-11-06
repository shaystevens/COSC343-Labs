__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "July 2022"

import numpy as np

def possible_moves(state, player):
   """
   Generates a list of states possible from the current state
   :param state: a list of rows (each a list of columns) of tictactoe board
   :param player: int indicating search for max (1) or min (-1) moves
   :return: a list of double lists of moves possible from state
   """

   state = np.array(state)
   possible_states = []
   for i in range(3):
      for j in range(3):
         if state[i][j] == 0:
            new_state = np.copy(state)
            new_state[i][j] = player
            possible_states.append(new_state.tolist())

   return possible_states


def maxs_possible_moves(state):
   """
   Generates a list of states possible from the current state
   with on max move
   :param state: a list of rows (each a list of columns) of tictactoe board
   :return: a list of board states
   """

   state = np.array(state)
   return possible_moves(state, player=1)


def mins_possible_moves(state):
   """Generates a list of states possible from the current state
   with on min move
   :param state: a list of rows (each a list of columns) of tictactoe board
   :return: a list of board states
   """

   return possible_moves(state, player=-1)


def remove_symmetries(states):
   """
   Removes rotational and mirror symmetric boards from a list of states
   :param states: a list board states
   :return: a list of board states (with symmetries removed)
   """

   # Rotation invariant states
   roti_states = [np.array(states[0])]

   # Examine all states
   for new_state in states:
      new_state = np.array(new_state)

      # Check for identical symmetries
      found_symmetry = False
      for rotation in [0, 1, 2, 3]:

         if rotation == 0:
            # Do not rotate
            ns = np.copy(new_state)
         else:
            # Rotate last state by 90 degrees
            ns = np.array([[ns[2, 0], ns[1, 0], ns[0, 0]],
                           [ns[2, 1], ns[1, 1], ns[0, 1]],
                           [ns[2, 2], ns[1, 2], ns[0, 2]]])

         for s in roti_states:
            # Check identical states
            if np.sum(s.reshape(-1) == ns.reshape(-1)) == 9:
               found_symmetry = True
               break

            # Check horizontal mirror
            if np.sum(s[:, 0] == ns[:, 2]) == 3 and np.sum(s[:, 2] == ns[:, 0]) == 3 and np.sum(
                    s[:, 1] == ns[:, 1]) == 3:
               found_symmetry = True
               break

            # Check vertical mirror
            if np.sum(s[0, :] == ns[2, :]) == 3 and np.sum(s[2, :] == ns[0, :]) == 3 and np.sum(
                    s[1, :] == ns[1, :]) == 3:
               found_symmetry = True
               break

         if found_symmetry:
            break

      # If new_state has a new symmetry, add it to the list of rotation invariant states
      if not found_symmetry:
         roti_states.append(new_state)

   # Convert to lists
   selected_states = []
   for s in roti_states:
      selected_states.append(s.tolist())

   # Return rotation invariant states
   return selected_states


def count_wins(state):
   """
   Count
   :param state:
   :return:
   """

   state = np.array(state)
   wins = 0
   for i in range(3):
      s = np.sum(state[i, :])
      if np.abs(s) == 3:
         wins += np.sign(s)

      s = np.sum(state[:, i])
      if np.abs(s) == 3:
         wins += np.sign(s)

   s = state[0, 0] + state[1, 1] + state[2, 2]
   if np.abs(s) == 3:
      wins += np.sign(s)

   s = state[2, 0] + state[1, 1] + state[0, 2]
   if np.abs(s) == 3:
      wins += np.sign(s)

   return wins


def terminal(state):
   state = np.array(state)
   if count_wins(state) == 0:
      for i in range(3):
         for j in range(3):
            if state[i, j] == 0:
               return False

   return True


def moves_left(state):
   if count_wins(state) != 0:
      return 0

   moves = 0
   for i in range(3):
      for j in range(3):
         if state[i, j] == 0:
            moves += 1

   return moves


def evaluate(state):
   true_wins = count_wins(state)
   if true_wins != 0:
      return np.sign(true_wins) * 100

   agent_state = np.copy(state)
   for i in range(3):
      for j in range(3):
         if agent_state[i, j] == 0:
            agent_state[i, j] = 1

   I_agent = count_wins(agent_state)

   opponent_state = np.copy(state)
   for i in range(3):
      for j in range(3):
         if opponent_state[i, j] == 0:
            opponent_state[i, j] = -1
   I_opponent = count_wins(opponent_state)

   return I_agent + I_opponent

def state_change_to_action(s_start,s_end):
   R,C = np.shape(s_start)

   if type(s_start)==np.ndarray:
      s_start = s_start.tolist()

   if type(s_end)==np.ndarray:
      s_end = s_end.tolist()

   for r in range(R):
      for c in range(C):
         if s_start[r][c] != s_end[r][c]:
            return (r,c)

