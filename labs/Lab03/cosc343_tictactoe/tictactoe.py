__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "July 2022"

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import os, sys
import importlib
import traceback
from settings import game_settings

# Class player holds all the agents for a given player
class Player:
   def __init__(self, playerFile,h):
      self.playerFile = playerFile
      self.errorMsg = ""
      self.ready = False

      if not os.path.exists(playerFile):
         print("Error! Agent file '%s' not found" % self.playerFile)
         sys.exit(-1)


      if len(playerFile) > 3 and playerFile[-3:].lower() == '.py':
         playerModule = playerFile[:-3]
      else:
         print("Error! Agent file %s needs a '.py' extension" % self.playerFile)
         sys.exit(-1)


      try:
         self.exec = importlib.import_module(playerModule)
      except Exception:
         print("Error! Failed to load '%s'" % self.playerFile)
         traceback.print_exc()
         sys.exit(-1)

      try:
         self.agent = self.exec.TicTacToeAgent(h=h)
      except Exception:
         print("Error! Failed to instantiate TicTacToeAgent() from '%s'" % self.playerFile)
         traceback.print_exc()
         sys.exit(-1)

class TicTacToe:

    def __init__(self, player1File, player2File):
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")
        self.plot_handles = []

        self.marks = ['o','x']

        self.reset()
        self.show()
        plt.ion()
        plt.show()


        player1 = Player(playerFile=player1File,h=self.h)
        player2 = Player(playerFile=player2File,h=self.h)

        self.players = [player1, player2]


    def reset(self):
        self.state = np.zeros((3,3)).astype('int8')

    def run(self):

        while True:

            for i,t in enumerate([1,-1]):

                self.h.set_title('%s\'s (%s) move...' % (self.players[i].playerFile,self.marks[i]))
                plt.pause(0.01)
                time.sleep(0.01)

                percepts = (np.copy(self.state)*t).tolist()
                try:
                    (r,c) = self.players[i].agent.AgentFunction(percepts)
                except:
                    print("Error! Failed to execute AgentFunction from '%s'" % self.players[i].playerFile)
                    traceback.print_exc()
                    sys.exit(-1)

                if type(r) != int or type(c) != int:
                    print("Error! The AgentFunction in '%s' must return an type of (int, int) type!" % self.players[i].playerFile)
                    traceback.print_exc()
                    sys.exit(-1)

                if r < 0 or r >= 3 or c<0 or c >= 3:
                    print("Error! The (%d, %d) values returned by AgentFunction in '%s' are out of bounds - must be values from set {0,1,2}!" % (r,c,self.players[i].playerFile))
                    traceback.print_exc()
                    sys.exit(-1)

                if self.state[r,c] != 0:
                    print("Error! The (%d, %d) moves returned by AgentFunction in '%s' is not an empty slot!" % (r,c,self.players[i].playerFile))
                    traceback.print_exc()
                    sys.exit(-1)

                self.state[r,c] = t
                self.plot_handles.append(
                    self.h.text(c + 0.5, 2 - r + 0.5, self.marks[i], fontsize=40, verticalalignment='center',
                                horizontalalignment='center'))
                self.h.figure.canvas.draw()
                plt.pause(0.1)
                time.sleep(0.1)

                if self.check_win() != 0:
                    plt.ioff()
                    plt.show()
                    return


    def check_win(self):
        winner = 0
        for i in range(3):
            s = np.sum(self.state[i, :])
            if np.abs(s)==3:
                winner = np.sign(s)
                self.plot_handles.append(self.h.plot( [0, 3],[2-i+0.5, 2-i+0.5], 'r'))
                break

            s = np.sum(self.state[:,i])
            if np.abs(s)==3:
                winner = np.sign(s)
                self.plot_handles.append(self.h.plot([i+0.5, i+0.5],[0, 3], 'r'))
                break

        s = self.state[0,0]+self.state[1,1]+self.state[2,2]
        if np.abs(s)==3:
            winner = np.sign(s)
            self.plot_handles.append(self.h.plot([3, 0], [0, 3], 'r'))

        s = self.state[2,0]+self.state[1,1]+self.state[0,2]
        if np.abs(s)==3:
            winner = np.sign(s)
            self.plot_handles.append(self.h.plot([0, 3], [0, 3], 'r'))

        if (winner < 0):
            plt.title("%s (%s) WON!" % (self.players[1].playerFile, self.marks[1]))
        elif winner != 0:
            plt.title("%s (%s) WON!" % (self.players[0].playerFile, self.marks[0]))

        if winner==0:
            draw = True
            for i in range(3):
                for j in range(3):
                    if self.state[i,j]==0:
                        draw=False

            if draw:
                plt.title("DRAW!")
                winner = 1

        if winner != 0:
            plt.pause(0.01)
            time.sleep(0.01)


        return winner

    def show(self):
        if not self.plot_handles:
            fh = plt.figure(figsize=(6, 6), dpi=100)
            self.h = fh.add_subplot(1, 1, 1)

            for x in range(1,3):
                self.h.plot([x, x], [0, 3], 'k')

            for y in range(1,3):
                self.h.plot([0, 3], [y, y], 'k')
            self.h.get_xaxis().set_visible(False)
            self.h.get_yaxis().set_visible(False)
            self.h.set_xlim([0, 3])
            self.h.set_ylim([0, 3])
            self.h.axis('off')


if __name__ == "__main__":

   game = TicTacToe(player1File=game_settings['player1File'],
                    player2File=game_settings['player2File'])

   game.run()

   game = TicTacToe(player1File=game_settings['player2File'],
                    player2File=game_settings['player1File'])

   game.run()

