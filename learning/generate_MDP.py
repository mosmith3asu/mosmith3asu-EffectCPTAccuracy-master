import numpy as np
from enviorment.worlds import WORLDS
from enviorment.utils import *
from data_tools.file_manager import save
import itertools
from tqdm import tqdm
iworld = 1
ixy = [0,1]
MDP_settings = {}
MDP_settings['pursuer'] = {}
MDP_settings['evader'] = {}

MDP_settings['pursuer']['Rcatch'] = 20
MDP_settings['pursuer']['pen'] = -3
MDP_settings['pursuer']['penProb'] = 0.5

MDP_settings['evader']['Rcatch'] = -20
MDP_settings['pursuer']['Rdist'] = -3 # scale on [0,1] = [min,max]

def newMDP(world = 1):
    """
    LOCATION \in [r,c]
    :param iworld:
    :return:
    """
    global iworld

    FILENAME = f'MDP_W{iworld}'

    iworld = world
    world = WORLDS[iworld]['array']
    borderVal = WORLDS['border_val']
    noPen, recPen = 0, 1  # whether or not penaly was recieved

    # MDP Def -----------------------------
    Sk = None # agent state
    A = None # joint action
    Tk = None # agent transition
    Rk = None # agent reward

    # Define State Space ---------------------------
    locs = np.array(np.where(world != borderVal),dtype='int8').T  # agent locations
    locs = np.append(locs,noPen * np.ones([np.shape(locs)[0],1],dtype='int8'),axis=1) # append no penalty states

    locs_2pen = [] # locations with 2 entries for penalty outcomes
    for r,c,pen in locs:
        if is_pen([r,c]): locs_2pen += [[r, c, noPen],[r, c,recPen]]
        else: locs_2pen += [[r, c,pen]] # else only 1 location
    Sk = np.array(list(itertools.product(*[np.array(locs_2pen),locs,locs]))) # <===

    # Define State space ---------------------------
    actions = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])  # [ wait, up, right,down, left]
    actions = np.append(actions, noPen * np.ones([np.shape(actions)[0], 1], dtype='int8'), axis=1) # apppend no penalty action

    A = np.array(list(itertools.product(*[actions,actions]))) # <===

    # Define Transitions -----------------------------
    wait = np.array([actions[0]])
    nSk = np.shape(Sk)[0]
    nA = np.shape(A)[0]
    Tk = np.zeros([nSk,nA,nSk])
    Rk = np.zeros(nSk)

    rCatch = MDP_settings['pursuer']['Rcatch']
    pPen = MDP_settings['pursuer']['penProb']
    rPen = MDP_settings['pursuer']['pen']
    for si in tqdm(range(nSk)):
        statei = Sk[si]

        # -- Rewards --
        ri = rCatch * int(is_caught(statei[:, ixy]))
        rhoi = rPen * statei[0, -1]
        try: Rk[si] = ri + rhoi
        except:
            pass
        # -- Transition --
        for ai,actioni in enumerate(A):
            statej = statei + np.append(actioni,wait,axis=0)
            sj = find_state(statej[:,ixy],Sk[:,:,ixy],raiseErr=False)
            if sj is not None: # valid state
                if np.size(sj) ==2:  Tk[si,ai,sj] = [pPen,1-pPen] # stochastic penalty
                else: Tk[si,ai,sj] = 1 # deterministic


    save_data = {}
    save_data['Sk'] = np.array(Sk,dtype='int8')
    save_data['A'] =  np.array(A,dtype='int8')
    save_data['Rk'] = np.array(Rk,dtype='float16')
    save_data['Tk'] = np.array(Tk,dtype='float16')
    save(save_data,FILENAME)

if __name__=="__main__":
    newMDP()
