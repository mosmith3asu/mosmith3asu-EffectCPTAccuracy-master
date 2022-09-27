"""
Centralized Learning Decentralized Execution

"""
import numpy as np

from enviorment.utils import *
from data_tools.file_manager import save,load
import itertools
from tqdm import tqdm
from agents.evader import EvaderDM
# from agents.robot import RobotDM
# from agents.evader import HumanDM
from enviorment.env import Enviorment

def epsilon_greedy(si,Q,eps=0.7):
    nA = Q.size()[2]
    if np.random.rand(1) <= eps:
        ai = np.argmax(Q[si])
    else:
        np.random.choice(np.arange(nA))

def QLearning(iworld = 1,num_episodes = 1000,learning_rate=0.6,discount=0.9):
    # Unpack MDP
    loaded = load(f'MDP_W{iworld}.npz')
    Sk =  loaded['Sk']
    A =   loaded['A']
    Rk =  loaded['Rk']
    Tk =  loaded['Tk']

    MDP = {'S': Sk,'A':A,'R':Rk,'T':Tk}

    nAgents = 2

    # Calc stats
    nSk = np.shape(Tk)[0]
    nA = np.shape(Tk)[1]

    # Set up agent policies
    eDM = EvaderDM(Sk)
    # hDM = HumanDM()
    # rDM = RobotDM()

    actions = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])  # [ wait, up, right,down, left]



    A = np.array([[0,0],[0,-1],[0,],[0,0],[0,0]])
    # Set up learning environment
    env = Enviorment(iworld,Sk,A,Rk,Tk)
    Q = np.zeros(Tk.shape()[0:2])
    # Q0 = initQ(nSk,nA,k=0,bias='dist',MDP = MDP) # <==== TEST =============================
    # Q1 =  initQ(nSk,nA,k=1,bias='dist',MDP = MDP) # <==== TEST =============================
    alpha = learning_rate
    gamma = discount
    nMoves = 20 # maximum number of moves in an iteration



    for epi in range(num_episodes):
        si = env.reset()
        for t in range(nMoves):

            ai = epsilon_greedy(si,Q)

            sj,rj,done = env.step(si,ai)

            # Calculate next action
            # next_sj = []
            # Rje = np.zeros(5)
            # Sje = []
            # Aj = []
            # for ai, action in enumerate(actions):
            #     statej = Sk[si] + np.array([[0, 0], [0, 0], action])
            #     sje = np.where(Sk == statej, axis=(1, 2)).flatten()[0]
            #     Sje.append(sj)
            #     Aje.append(Q[sj])
            #     Rje[ai] = eDM.get_reward(sj)
            #
            # pde = eDM.get_pd(Rje)
            # aj = [np.argmax(Q[sj])]



            # TD update
            aj = np.argmax(Q[sj])
            td_target = rj + gamma * Q[sj][aj]
            td_delta = td_target - Q[si][ai]
            Q[si][ai] += alpha * td_delta

            if done: break


            # Move Evader
            for ai, action in enumerate(actions):
                statej = Sk[si] + np.array([[0, 0], [0, 0], action])
                sje = np.where(Sk == statej, axis=(1, 2)).flatten()[0]
                Sje.append(sj)
                Aje.append(Q[sj])
                Rje[ai] = eDM.get_reward(sj)


            si = sj
















if __name__ == "__main__":
    QLearning()
