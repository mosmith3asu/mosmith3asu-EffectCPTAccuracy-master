from enviorment.worlds import WORLDS
import numpy as np
from enviorment.utils import *
# from learning.MDP.MDP_settings import *
# import matplotlib.pyplot as plt
from math import dist
import random
random.seed(0)
class Enviorment(object):
    def __init__(self,iworld,S,A,R,T):
        # self.MDP = MDP
        self.iworld = iworld
        self.A = A
        self.S = S
        self.R = R
        self.T = T
        self.nA = np.shape(A)[0]
        self.nS = np.shape(S)[0]

        self.r_caught = 20
        self.r_penalty = 3
        self.p_penalty = 0.5

        self.start = np.array(WORLDS[iworld]['start'])/100
        self.world = WORLDS[iworld]['array']
        self.si_start = np.where(np.all(S==self.start,axis=(1,2)))[0][0]
        self.round,self.max_rounds = 0,20

        self.was_caught = 0
        self.init_params = {}
        self.init_params['round'] = self.round
        self.init_params['max_rounds'] = self.max_rounds
        self.init_params['was_caught'] = self.was_caught

        self.si = self.reset()

    def step(self,si,ai):
        sj = np.random.choice(np.arange(self.nS),p=self.T[si,ai,:])

        self.round += 1
        caught = is_caught(self.S[sj])
        penalty = is_pen(self.S[sj])

        done = False
        done = True if caught else done
        done = True if self.round >= self.max_rounds else done

        rewards = [0,0]
        for player in [0,1]:
            rewards[player] += self.r_caught - self.round if caught else 0
            rewards[player] += np.random.choice([0,self.r_penalty],p=[1-self.p_penalty,self.p_penalty]) if penalty[player] else 0

        return sj,rewards,done

    def reset(self):
        for key in self.init_params: self.__dict__[key] = self.init_params[key]
        return random.choice(np.arange(self.nS))

    # def take_joint_pursuer_action(self,si, ai):
    #     if self.is_done(si): return np.array([0,0]) , si#"done"
    #     evader = 2
    #
    #     # get explicit space
    #     statei = self.S[si]
    #     actioni = self.A[ai]
    #     statej = np.copy(statei)
    #     statej[0:2,:] += actioni
    #
    #     # get admissable enxt states
    #     sj1_adm = np.where(self.T[si, ai, :] > 0)[0]
    #     sj2_adm = np.where(np.all(self.MDP.joint.S[:, evader, :] == statei[evader, :], axis=1))[0]
    #     try: sj = sj1_adm[np.where([(sj1 in sj2_adm) for sj1 in sj1_adm])[0][0]]
    #     except: raise Exception("Cannot find common sj")
    #
    #     # Get reward
    #     reward = np.copy(self.R[sj])
    #     # subtract time penalty from catching target
    #     # reward = np.array([max(0,reward[player]-self.round) for player in [0,1]]) if is_caught(self.MDP.joint.S[sj]) else reward
    #     if is_caught(self.S[sj]):
    #         reward = reward - self.round
    #         # print(f'TESTING CAUGHT')
    #     return reward, sj
