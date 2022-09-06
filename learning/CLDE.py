"""
Centralized Learning Decentralized Execution

"""
from enviorment.utils import *
from data_tools.file_manager import save,load
import itertools
from tqdm import tqdm
from agents.evader import EvaderDM
from agents.robot import RobotDM
from agents.evader import HumanDM
from enviorment.env import Enviorment

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
    hDM = HumanDM()
    rDM = RobotDM()


    # Set up learning environment
    env = Enviorment(iworld,Sk,A,Rk,Tk)
    Q0 = initQ(nSk,nA,k=0,bias='dist',MDP = MDP) # <==== TEST =============================
    Q1 =  initQ(nSk,nA,k=1,bias='dist',MDP = MDP) # <==== TEST =============================
    alpha = learning_rate
    gamma = discount
    nMoves = 20 # maximum number of moves in an iteration



    for epi in range(num_episodes):
        si = env.reset()
        for t in range(nMoves):

            pH = hDM.












if __name__ == "__main__":
    Qlearning()
