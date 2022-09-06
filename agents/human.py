# import numpy as np
# import matplotlib.pyplot as plt
import numpy as np
from data_tools.file_manager import save,load
from enviorment.utils import *
from agents.policies import CPT_Handler,noisy_rational
import itertools
class HumanDM(object):
    def __init__(self,Sk,A,Rk,Tk,iagent = 1):
        self.idx = iagent
        self.Sk = Sk
        self.A = A
        self.Rk = Rk
        self.Tk = Tk
        self.nA = np.shape(self.Tk)[1]
        self.nk = np.shape(self.A)[1] # number of agents io joint action
        self.nAk = 5 # number of indepedant agent actions
        self.Ai =np.array(list(itertools.product(*[np.arange(self.nAk),np.arange(self.nAk)])))
        # print(self.Ai)

        self.bounds = {}
        self.bounds['b'] = [0,0] # reference point
        self.bounds['gamma'] = [0.2, 0.8] # diminishing return gain
        self.bounds['lam'] = [0.01, 10] # loss aversion
        self.bounds['alpha'] = [0.01, 10] # prelec parameter
        self.bounds['delta'] = [0.2, 0.8] # convexity
        self.bounds['theta'] = [1, 1] # rationality
        self.bounds['resolution'] = 10
        self.new_params()
        self.DM = CPT_Handler(self.b,self.gamma,self.lam,self.alpha,self.delta,self.theta)

        print(f'\nHuman DM initialized...')
    def new_params(self):
        self.b = np.random.choice(np.linspace(self.bounds['b'][0],self.bounds['b'][1],self.bounds['resolution']))
        self.gamma = np.random.choice(np.linspace(self.bounds['gamma'][0],self.bounds['gamma'][1],self.bounds['resolution']))
        self.lam =  np.random.choice(np.linspace(self.bounds['lam'][0],self.bounds['lam'][1],self.bounds['resolution']))
        self.alpha = np.random.choice(np.linspace(self.bounds['alpha'][0],self.bounds['alpha'][1],self.bounds['resolution']))
        self.delta = np.random.choice(np.linspace(self.bounds['delta'][0],self.bounds['delta'][1],self.bounds['resolution']))
        self.theta = np.random.choice(np.linspace(self.bounds['theta'][0],self.bounds['theta'][1],self.bounds['resolution']))

        # DM = CPT_Handler(self.b, self.gamma, self.lam, self.alpha, self.delta, self.theta)
        # self.Rk_perc = self.DM.utility_weight(self.Rk)
    def print_params(self):
        print(f'b     = {round(self.b    ,2 )}')
        print(f'gamma = {round(self.gamma,2 )}')
        print(f'lam   = {round(self.lam  ,2 )}')
        print(f'alpha = {round(self.alpha,2 )}')
        print(f'delta = {round(self.delta,2 )}')
        print(f'theta = {round(self.theta,2 )}')
    def get_reward(self,statei):
        if self.idx == 0:  # lead agent
            if np.size(statei) == 1: si = statei  # index given
            else: si = find_state(statei[:, 0:2], self.Sk[:, :, 0:2])  # full state given
        elif self.idx ==1: # not lead agent
            si, statei = switch_P1P2_state(statei,self.Sk)
        else: raise Exception('Unknown agent index in HumanDM')
        return self.Rk[si]
    def get_pd(self,si, R,pA_notk,get_joint=False):
        """ pai is probability across all joint actions """
        Eaik = np.zeros(5)  # expacted value of agent actions

        # Switch state
        if self.idx == 1:
            si, statei = switch_P1P2_state(si,self.Sk)
            try: si = si[0] # default to singular non-penalty state
            except: pass

        # Switch probabilities (put uncertainty in partner idx=1)
        pA = np.ones([self.nA, self.nk])
        if self.idx == 1:  # lead player is uncertain
            for aik in range(self.nAk):
                iold = np.where(self.Ai[:, 0] == aik)
                inew = np.where(self.Ai[:, 1] == aik)
                pA[inew,1] = pA_notk[iold]
        else:  pA[:,1] = pA_notk # not lead player is uncertain

        # Utility Transformation ----------------
        R_perc = self.DM.utility_weight(R)

        # Probability Transformation and Expected Value --------
        aik_adm = np.zeros(self.nAk,dtype='int8')
        for ai in range(self.nA):
            aik = self.Ai[ai,0]                                         # non-joint action index
            p_k = pA[ai,0]                                              # probability of ego transition (p=1)
            p_notk = pA[ai,1]                                           # probability of partner transition
            sj_adm = np.array(np.where(self.Tk[si,ai,:]>0)).flatten()   # non zero transitions
            for sj in sj_adm:
                p_env = self.Tk[si, ai, sj]         # probably of uncertain environment trans (penalty)
                p_sj = (p_k) * (p_notk) * (p_env)   # probability of next state
                p_sj = self.DM.prob_weight(p_sj)    # apply H bias on probability
                Eaik[aik] += p_sj*R_perc[ai]        # sum up expected value
                aik_adm[aik] = 1                    # found admissible action

        # Policy on expected value ----------
        aik_adm = np.where(aik_adm==1)
        pdAk = np.zeros(self.nAk)
        pdAk[aik_adm] = noisy_rational(Eaik[aik_adm],self.theta)
        if not get_joint: return pdAk

        # Convert to joint action probs --------
        pdA = np.zeros(self.nA)
        for aik in range(self.nAk):  pdA[np.where(self.Ai[:,self.idx] == aik)] = pdAk[aik]
        return pdA
    def choose(self,si, Qsi,pA_notk,jointly=False):
        pdA = self.get_pd(si, Qsi, pA_notk, get_joint=jointly)
        if not jointly: return np.random.choice(np.arange(self.nAk), p=pdA)
        else: return np.random.choice(np.arange(self.nA), p=pdA)



if __name__ == "__main__":
    loaded = load(f'MDP_W{iworld}.npz')
    Sk = loaded['Sk']
    A = loaded['A']
    Rk = loaded['Rk']
    Tk = loaded['Tk']
    nAk=5
    print(f'Starting human.py')
    # nS = 11025
    # nA = 25
    # Sk =np.zeros([nS,3,3])
    # A = np.zeros([nA,2,3])
    # Rk = np.zeros(nS)
    # Tk = np.zeros([nS,nA,nS])
    hDM = HumanDM(Sk,A,Rk,Tk)

    si = 0
    Qsi = np.arange(25)
    pdAhat_robot = np.array(list(itertools.product(*[np.arange(nAk), np.arange(nAk)])))[:,0]
    pdAhat_robot = pdAhat_robot/np.sum(pdAhat_robot)
    # pd = hDM.get_pd(si,Qsi,pA_notk=pdAhat_robot)
    print([hDM.choose(si, Qsi,pdAhat_robot) for i in range(10)])
