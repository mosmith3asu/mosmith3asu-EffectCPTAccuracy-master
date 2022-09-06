import math
import warnings

from learning.generate_MDP import iworld
from enviorment.worlds import WORLDS
import numpy as np
# import matplotlib.pyplot as plt


def find_state(state,S,raiseErr=True):
    si = np.array(np.where(np.all(S==state,axis=(1,2)))).flatten()
    if len(si)==1: si = si[0]
    elif len(si)==0:
        if raiseErr: raise Exception(f'find_state\n>>{state}\n\t>> STATE NOT FOUND')
        else: si = None
    return si
def find_action(action,A):
    try: ai = np.array(np.where(np.all(A == action, axis=(1, 2)))).flatten()
    except: ai = np.array(np.where(np.all(A==action,axis=1))).flatten()
    if len(ai)==1: ai = ai[0]
    elif len(ai)==0: raise Exception(f'find_action{action}>> ACTION NOT FOUND')
    return ai
def switch_P1P2_state(statei,Sk):
    if np.size(statei) == 1:   statei = np.copy(Sk[statei]) # index given
    statei = np.array([statei[1], statei[0], statei[2]])
    si = find_state(statei[:, 0:2], Sk[:, :, 0:2])
    return si,statei
def switch_P1P2_action(actioni,A):
    if np.size(actioni) == 1: actioni = np.copy(A[actioni]) # index give
    actioni = np.array([actioni[1], actioni[0]])
    ai = find_action(actioni, A)
    return ai,actioni
######################################################################################
################## state check functions #############################################
######################################################################################
def is_border(state, S=None):
    if S is not None: state = S[state]  # assume that state is explicity locations
    nAgents = min(2, int(np.size(state) / 2))
    if nAgents == 1: state = np.array(state).reshape([1, 2])
    else:  state = np.array(state)[:nAgents].reshape([2, 2])
    checkVal = WORLDS['border_val']
    result =[WORLDS[iworld]['array'][state[k][0],state[k][1]] == checkVal for k in range(nAgents)]
    if nAgents==1: result = result[0]
    return result
def is_pen(state,S=None):
    if S is not None: state = S[state] # assume that state is explicity locations
    nAgents = min(2,int(np.size(state) / 2))
    if nAgents ==1: state = np.array(state).reshape([1,2])
    else: state = np.array(state)[:nAgents].reshape([2, 2])
    checkVal = WORLDS['pen_val']
    result =[WORLDS[iworld]['array'][state[k][0],state[k][1]] == checkVal for k in range(nAgents)]
    if nAgents==1: result = result[0]
    return result
def is_caught(state,S=None):
    if S is not None: state = S[state] # assume that state is explicity locations
    isAdjacent0 = (math.dist(state[0],state[2]) <= 1)
    isAdjacent1 = (math.dist(state[1],state[2]) <= 1)
    if isAdjacent0 and isAdjacent1: caught = True
    else: caught = False
    return caught
######################################################################################
################## learning tools ####################################################
######################################################################################

def initQ(nS,nA,k,bias = None,MDP=None,R_scale=1.0):
    """
    ############### DOES THIS NEED TO BE INVERTED? ##############
    """
    def r_dist(statei,k,rmax = None,from_k = 2):
        if rmax is None: return math.dist(statei[k, 0:2], statei[from_k, 0:2])
        else: return (rmax-math.dist(statei[k, 0:2], statei[from_k, 0:2]))/rmax


    if bias is None: Q = np.zeros([nS,nA])
    else:
        Q = np.zeros([nS, nA])
        state_maxdist = np.array([[0,0,0],[0,0,0],[5,5,0]])
        rmax = r_dist(state_maxdist,k)
        noMove = np.array([[0,0,0]])
        for si, statei in enumerate(MDP['S']):
            for ai, actioni in enumerate(MDP['A']):
                qsa = 0
                statej = np.append(statei + actioni, noMove, axis=0)
                if bias == 'dist' or 'dist' in bias:                # small incentive to get closer
                    qsa += R_scale * r_dist(statej, k, rmax=rmax)
                # if bias == 'reward' or 'reward' in bias:            # payoff for catching
                #     sj = find_state(statej,S=MDP['S'])              # get state index
                #     qsa += np.sum(MDP['T'][si,ai,sj]*MDP['R'][sj])  # expected reward from joint action
                Q[si,ai] = qsa
    return Q