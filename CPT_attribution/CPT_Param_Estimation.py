import itertools
# import matplotlib.pyplot as plt
from learning.MDP.MDP_settings import *
from learning.Qlearning.Qlearning_Cent_utils import load_Qfun
from learning.MDP.MDP_tools import load_MDP
import numpy as np
import nashpy as nash
from functools import partial
from itertools import product
from enviorment.assets.worlds import WORLDS
from learning.Qlearning.Qlearning_Cent_utils import *
from learning.learning_utils.enviorment_utils import Enviorment
from learning.Qlearning.Qlearning_Cent_utils import initQ_evader
from scipy.optimize import minimize
import scipy.stats as stats
from pandas import DataFrame
from learning.MDP.MDP_settings import MDP_file_path
from decision_models.CPT_prelec import CPT
# MDP_path ='C:\\Users\\mason\\Desktop\\Effects_of_CPT_Accuracy\\learning\\MDP\\Cent_MDP_data'
MDP_path =f'C:\\Users\\mason\\Desktop\\Effects_of_CPT_Accuracy\\learning\\MDP\\{MDP_file_path}'
loaded = np.load(MDP_path + '.npz')
S, A, T_int, R_player, R_evader = loaded['S'], loaded['A'], loaded['T'], loaded['R_player'], loaded['R_evader']
# A = np.array(list(product(ACTIONS['list'],  ACTIONS['list'])), dtype='int8')
nS = np.shape(S)[0]
Ak = ACTIONS['list']
nA = np.shape(A)[0]
nAk = np.shape(Ak)[0]
BOUNDS01 = (0.2,0.8)
BOUNDS0inf = (0.2,10)


def NR_decision(V, rationality, choose=False, admissable=None):
    nActions = np.size(V)
    pd = np.zeros(np.shape(V))
    if admissable is None: admissable = np.arange(nActions)
    iadm = admissable
    Vadm = np.copy(V[iadm])
    pd[iadm] = np.exp(rationality * Vadm) / np.sum(np.exp(rationality * Vadm))
    if choose: return np.random.choice(np.arange(len(pd)), p=pd)
    else: return pd


def main(num_episodes=7,  iworld = 1):
    #######################################################
    # INIT ENVIORMENT #####################################

    player, partner,evader = 0, 1,2
    evader_rationality = 10

    # file_name = 'Qfunction_Cent'
    # file_name = f'Qfunction_Cent_W{iworld}'
    file_name = f'Qfunction_Cent_8_24_22'
    Q, stats = load_Qfun(file_name)

    MDP = MDP_DataClass(iworld, MDP_path=MDP_path)
    env = Enviorment(iworld, MDP)
    obs = {}
    obs['si'] = []
    obs['ai'] = []
    obs['pd'] = []

    stats_init = {}
    stats_init['epi_reward'] = []
    stats_init['epi_length'] = []
    stats_init['epi_caught'] = []
    stats_init['epi_psucc'] = []
    stats = Struct(**stats_init)
    #######################################################
    # INIT HUMAN AND EVADER PLAYER ########################
    H_CPT_params = init_H_params()
    Human_DM = partial(humanDM, Q=Q, CPT_params=H_CPT_params, bounds=BOUNDS01)
    Robot_DM = partial(robotDM, Q=Q)
    Evader_DM =partial(noisy_rational,rationality=evader_rationality)


    print('\n\n###########################')
    print('#### H Params #############')
    print('###########################')
    for key in H_CPT_params: print(f'\t| {key}={H_CPT_params[key]}')


    print('\n\n###########################')
    print('### INIT EVADER Q-FUN #####')
    print('###########################')
    Q_evd = initQ_evader(MDP)

    #######################################################
    # START ###############################################
    si = env.reset()
    print('\n\n###########################')
    print('#### BEGIN PLAY ###########')
    print('###########################')
    for ith_episode in range(num_episodes):
        if env.was_caught: print(f'\t EVADER CAUGHT')
        stats.epi_length.append(0)
        stats.epi_caught.append(env.was_caught)
        stats.epi_psucc.append(100 * np.mean(stats.epi_caught) if np.size(stats.epi_caught) > 8 else 0)

        si = env.reset()
        for t in itertools.count():
            # Get Choose Actions ############
            # ToM = 1 --------------------
            robot_pd = Robot_DM(si=si, pd_partner='uniform')
            human_pd = Human_DM(si=si, pd_partner='uniform')

            # ToM = 2 --------------------
            # est_robot_pd = Robot_DM(si=si, pd_partner='uniform')
            # est_human_pd = Human_DM(si=si, pd_partner='uniform')
            # human_pd = Human_DM(si=si, pd_partner=est_robot_pd)
            # robot_pd = Robot_DM(si=si, pd_partner=est_human_pd )
            robot_iak = np.random.choice(np.arange(nAk),p=robot_pd)
            human_iak = np.random.choice(np.arange(nAk), p=human_pd)
            obs['si'].append(si), obs['ai'].append(human_iak),obs['pd'].append(human_pd) # make observation


            actioni = np.array([Ak[robot_iak],Ak[human_iak]])

            # ia = action2joint_idx(actioni)
            statej=np.copy(S[si])
            statej[0:2] += actioni
            sj = state2joint_idx(statej)

            report = '\r'
            report += f'\t| epI = {ith_episode}  t={t}  '
            report += f'\tstate = {[list(stateik) for stateik in MDP.joint.S[si]]} '
            report += f'\tP(Success) = {round(stats.epi_psucc[ith_episode])}% '
            print(report, end='')

            # Check Environment State ------------------
            is_bad_state = contains_border(WORLDS['empty_world']['array'], WORLDS['border_val'], statej)
            if is_bad_state: raise Exception(f"ERRR: [PURS Turn] State j is out of bounds... \n State j = {statej}")
            if env.is_done(sj): break

            ##########################################
            # EVADER TURN ############################
            Qsi_evd = Q_evd[sj]
            pd_evd = Evader_DM(Qsi_evd, admissable=np.where(Qsi_evd != 0))  # greater than zero trans prob)
            ai_evd = np.random.choice(np.arange(len(pd_evd)), p=pd_evd)
            # try: ai_evd = np.random.choice(np.arange(len(pd_evd)), p=pd_evd)
            # except Exception as inst: raise Exception( f'EVADER ERROR \n {inst} \n pd_evd={np.shape(pd_evd)} Q_evd[si_evd]={np.shape(Q_evd[sj])}')

            statej = np.copy(MDP.joint.S[sj])
            statej[evader] += MDP.agent.A[ai_evd]
            sj = np.where(np.all(MDP.joint.S == statej, axis=(1, 2)))[0][0]

            # Check Environment State ------------------
            is_bad_state = contains_border(WORLDS['empty_world']['array'], WORLDS['border_val'], statej)
            if is_bad_state: raise Exception(f"ERRR: [EVADER Turn] State j is out of bounds... \n State j = {statej}")
            if env.is_done(sj): break

            si = sj
            env.round += 1
        #### END OF EPI ######
    #### END OF LOOP ######

    print('\n\n###########################')
    print('#### MAX LIKELIHOOD #######')
    print('###########################')
    nObs = len(obs['si'])
    print(f'\t| running optimization [ns = {nObs}]..')
    params_hat = max_likelyhood_est(Q, obs['si'], obs['ai'])
    print(f'\t| results [nGames={num_episodes} nSamples = {nObs}]:')
    df_paramas_dict = {}
    for key in H_CPT_params:
        try: df_paramas_dict[f'| {key}'] = [np.round(H_CPT_params[key],2),np.round(params_hat[key],2),np.round(H_CPT_params[key]-params_hat[key],3)]
        except: df_paramas_dict[f'| {key}'] = [np.nan,np.nan,np.nan]
    df_params = DataFrame.from_dict(df_paramas_dict, orient='index',columns=['Truth', 'Est', 'Diff'])
    print(df_params)

    print(f'\t| testing fitted accuracy over nG={num_episodes}..')
    Human_DM_hat = partial(humanDM, Q=Q, CPT_params=params_hat, bounds=BOUNDS01)
    pdi_hat = [Human_DM_hat(si=obs['si'][t], pd_partner='uniform') for t in range(nObs)]

    # stats ---------------------
    mean_cum_pdiff = np.mean(np.array(obs['pd'])-np.array(pdi_hat))
    mean_obs_pdiff = np.mean([obs['pd'][t][obs['ai'][t]]- pdi_hat[t][obs['ai'][t]] for t in range(nObs)])
    mean_abs_cum_pdiff = np.mean(np.abs(np.array(obs['pd'])- np.array(pdi_hat)))
    mean_abs_obs_pdiff = np.mean(np.abs([obs['pd'][t][obs['ai'][t]] - pdi_hat[t][obs['ai'][t]] for t in range(nObs)]))
    Prob_M = np.prod([obs['pd'][t][ np.argmax(pdi_hat[t])] for t in range(nObs)])
    # Prob_M = np.sum(np.log([obs['pd'][t][np.argmax(pdi_hat[t])] for t in range(nObs)]))

    # report --------------------
    print(f'\t| fitted accuracy results:')
    print(f'\t\t| mean cumulative  P diff  = {round(mean_cum_pdiff,6)}')
    print(f'\t\t| mean cumulative |P diff| = {round(mean_abs_cum_pdiff, 6)}')
    print(f'\t\t| mean observed    P diff  = {round(mean_obs_pdiff, 6)}')
    print(f'\t\t| mean observed   |P diff| = {round(mean_abs_obs_pdiff, 6)}')
    print(f'\t\t| prob of model     P(M)   = {round(Prob_M,6)}')
    print(f'\t| Finished...')
def robotDM(Q, si, player=0, pd_partner='uniform'):
    def uniform(Vak):  return np.ones(nAk) / nAk
    Va =Q[si]
    pd_predict = np.empty(nA)
    pd_partner = uniform(0) if pd_partner == 'uniform' else pd_partner
    partner = 1 if player == 0 else 0

    # Partner Prob Dist over Joint Actions ------------------
    for iak, actionk in enumerate(Ak):
        ia_partner = action2joint_idx(actionk, player=partner)
        pd_predict[ia_partner] = pd_partner[iak]

    ##########################################
    # GET STATS ##############################
    i_adm = []
    EVal = np.empty(nAk)
    for iak_player, actionk_player in enumerate(Ak):
        ia = action2joint_idx(actionk_player, player=player)  # possible joint actions given player action (list)
        EVal[iak_player] = np.sum(Va[ia] * pd_predict[ia])
        if not np.all(T_int[si,ia,:]==0): i_adm.append(iak_player)
        # i_adm.append(np.where(Q[si,ia]))
    i_adm = np.unique(i_adm)
    # i_adm = np.array([1 in T_int[si,ai,:] for ai in ])
    # i_adm = np.array(np.where(EVal!=0)).flatten()
    pd = np.zeros(nAk)
    pd[i_adm[np.argmax(EVal[i_adm])]] = 1

    return pd



CPT_params ={}
CPT_params['b'] = None # reference
CPT_params['gamma']   = 0.5 # utility gain
CPT_params['lam']     = 5 # relative weighting of gains/losses
CPT_params['alpha']   = 0.5 # prelec parameter
CPT_params['delta']   = 0.5 # convexity gain?
CPT_params['theta'] = 10 # rationality
humanCPT = CPT(**CPT_params)

def humanDM(Q, si, CPT_params, bounds=BOUNDS01,player=1, pd_partner='uniform'):  # partnerDM=None

    humanCPT.update_params(**CPT_params)
    def uniform(Vak): return np.ones(nAk) / nAk

    Va = np.copy(Q[si])
    pd_predict = np.empty(nA)
    # pd_player = np.empty(nA)
    # pd_partner = partnerDM(Va)
    pd_partner = uniform(0) if pd_partner == 'uniform' else pd_partner
    partner = 1 if player == 0 else 0


    # Partner Prob Dist over Joint Actions ------------------
    for iak, actionk in enumerate(Ak):
        ia_joint = action2joint_idx(actionk, player=partner)
        pd_predict[ia_joint] = pd_partner[iak]

    EVal = np.empty(nAk)
    for iak_player, actionk_player in enumerate(Ak):
        ia = action2joint_idx(actionk_player, player=player)  # possible joint actions given player action (list)
        EVal[iak_player] = np.sum(Va[ia] * pd_predict[ia])
    i_adm = np.array(np.where(EVal != 0)).flatten()

    perc_r = humanCPT.utility_weight(Va) # percieved weight of losses and gains
    perc_pa = humanCPT.prob_weight(pd_predict) # percieved probability of outcome give action

    pec_EVal = np.empty(nAk)
    # i_adm = np.empty(nAk,dtype='int8')
    for iak_player, actionk_player in enumerate(Ak[i_adm]):
        ia = action2joint_idx(actionk_player, player=player)  # possible joint actions given player action (list)
        pec_EVal[iak_player] = np.sum(perc_r[ia] * perc_pa[ia])
    perc_pd = noisy_rational(pec_EVal,humanCPT.theta, admissable=i_adm)  # np.where(pec_EVal != 0)[0])
    perc_pd =np.nan_to_num(perc_pd)

    return perc_pd/np.sum(perc_pd)  # ,pec_EVal,EVal
    # # Reduce joint Q-fun to single agent -------------------
    # Sj_adm = admissible_next_states(si)
    # Qk_tmp = np.empty([nS,nAk])
    # for sj in np.where(Sj_adm==1)[0]:
    #     for iak_player, actionk_player in enumerate(Ak):
    #         ia = action2joint_idx(actionk_player, player=player)  # possible joint actions given player action (list)
    #         Qk_tmp[sj,iak_player] = np.sum(Q[sj,ia] * pd_predict[ia])
    #
    # # perc_r = humanCPT.utility_weight(Vak) # percieved weight of losses and gains
    # # perc_pa = humanCPT.prob_weight(pd_) # percieved probability of outcome give action
    #
    # T = np.zeros([nS, nAk, nS])
    # for iak0, actionk0 in enumerate(Ak): # for each ego action
    #     for iak1, actionk1 in enumerate(Ak):  # for each partner action
    #         statej =np.copy(S[si])
    #         statej[player] = actionk0
    #         statej[partner] = actionk1
    #         sj = state2joint_idx(statej)
    #         prob = pd_partner[iak1]
    #         T[si,iak0,sj] = prob
    #
    # # ichoice, perc_pd = humanCPT(Vak, Tsi)  # precieved pd
    # ichoice, perc_pd = humanCPT(Qk_tmp, T,si)  # precieved pd
    # # # ichoice, perc_pd = humanCPT(Q[si,:], Tsi) # precieved pd
    # return perc_pd



def max_likelyhood_est(Q,si_obs,ai_obs):
    """
    We can effectively ignore the prior and the evidence because — given the Wiki definition of a uniform prior
    distribution — all coefficient values are equally likely. And probability of all data values (assume continuous)
    are equally likely, and basically zero.
    maxLL docs: https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f
    scipy minimization docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    # posterior = likelihood x prior / evidence

    # Define Likelyhood Function ------------
    def CPT_handler(params,Q,si_obs,ai_obs,const,param_names):
        # UNPACK PARAMS
        CPT_params = {}
        for i,key in enumerate(param_names): CPT_params[key] = params[i] # add in design variables
        for key in const: CPT_params[key] = const[key] # add in constants

        # CALC PROBABILITY OF OBS ai UNDER MODEL
        nObs = len(si_obs)
        Pai_given_M = [humanDM(Q, si_obs[i], CPT_params)[ai_obs[i]] for i in range(nObs)]
        return  -np.sum(np.log(Pai_given_M))
        # return -np.prod(Pai_given_M)

        # Pai_given_M = np.array([humanDM(Q, si_obs[i], CPT_params) for i in range(nObs)])
        # return -np.sum(stats.norm.logpdf(ai_obs, loc=np.argmax(Pai_given_M,axis=1))) #, scale=sd

    param_names = ['gamma','lam','alpha','delta']
    guess_params = {}
    guess_params['gamma'] = 0.5  # utility gain
    guess_params['lam'] = 5  # relative weighting of gains/losses
    guess_params['alpha'] = 0.5  # prelec parameter
    guess_params['delta'] = 0.5  # convexity gain?

    const_params = {}
    const_params['b'] = None
    const_params['theta'] = 10  # rationality

    bounds_params = {}
    bounds_params['b'] = None
    bounds_params['gamma'] = BOUNDS01
    bounds_params['lam'] = BOUNDS0inf
    bounds_params['alpha'] = BOUNDS0inf
    bounds_params['delta'] = BOUNDS01
    bounds_params['theta'] = 10

    # Define opt vars -------------------
    SOLVER = 'Nelder-Mead'
    guess = np.array([guess_params[key] for key in guess_params])
    bounds = np.array([bounds_params[key] for key in guess_params])
    obj_fun = partial(CPT_handler,Q=Q,si_obs=si_obs,ai_obs=ai_obs,const=const_params,param_names = param_names)
    results = minimize(obj_fun, guess,bounds=bounds,method = SOLVER, options = {'disp': True})# method= 'Nelder - Mead'
    params0 = results['x']

    est_params = {}
    for i,key in enumerate(param_names):  est_params[key] = params0[i] # design variables
    for key in const_params:  est_params[key] = const_params[key]  # design variables
    return est_params

def admissible_next_states(si):
    Sj_adm = np.zeros(nS)
    for ai in range(nA):
        Sj_adm[np.where(T_int[si,ai,:]==1)] = 1
    return Sj_adm
def compute_NE(Va,nplayers=2,symmetrical=True):
    if symmetrical:  bimatrix = np.empty([nAk,nAk])
    else: bimatrix = np.empty([nAk,nAk,nplayers]) # number of actions for each player

    for ia,action in enumerate(A):
        iak = [action2player_idx(action[player]) for player in range(nplayers)]
        if symmetrical: bimatrix[iak[0],iak[1]] = Va[ia]
        else:
            bimatrix[iak[0], iak[1], 0] = Va[ia, 0]
            bimatrix[iak[0], iak[1], 1] = Va[ia, 1]

    equilibria = nash.Game(bimatrix).support_enumeration()
    for eq in equilibria: print(eq)
    return eq
def noisy_rational(V, rationality, choose=False, admissable=None):
    nActions = np.size(V)
    pd = np.zeros(np.shape(V))
    if admissable is None: admissable = np.arange(nActions)
    iadm = admissable
    Vadm = np.copy(V[iadm])
    pd[iadm] = np.exp(rationality * Vadm) / np.sum(np.exp(rationality * Vadm))
    if choose: return np.random.choice(np.arange(len(pd)), p= pd/np.sum(pd))
    else: return pd/np.sum(pd)
def state2joint_idx(state,player=None):
    if player is None and np.size(state)==2: player=1
    elif player is None and np.size(state)==4: player=[0,1]
    elif player is None and np.size(state) == 6:  player = [0, 1,2]

    if isinstance(player,int): ia = np.array(np.where(np.all(S[:,player,:] == state, axis=1))).flatten()
    else:  ia =np.array(np.where(np.all(S[:,player,:] == state, axis=(1,2)))).flatten() #elif np.size(player)==2:
    if np.size(ia)==1: ia = ia[0]

    return ia
def action2joint_idx(action,player=None):

    if player is None and np.size(action)==2: player=1
    elif player is None and np.size(action)==4: player=[0,1]
    elif player is None and np.size(action) == 6:  player = [0, 1,2]

    if isinstance(player,int): ia = np.array(np.where(np.all(A[:,player,:] == action, axis=1))).flatten()
    else:  ia =np.array(np.where(np.all(A[:,player,:] == action, axis=(1,2)))).flatten() #elif np.size(player)==2:
    if np.size(ia)==1: ia = ia[0]

    return ia
def action2player_idx(action):
    ia = np.array(np.where(np.all(Ak == action, axis=1))).flatten()
    if np.size(ia)==1: ia = ia[0]
    return ia
def init_H_params():
    np.random.seed(4)
    H_res = 1000
    #################################
    # INIT ESTIMATED AGENT ##########
    H_CPT_params = {}
    H_CPT_params['b'] = None  # reference
    H_CPT_params['gamma'] = np.random.choice(np.linspace(BOUNDS01[0], BOUNDS01[1], H_res))  # utility gain
    H_CPT_params['lam'] =  np.random.choice(np.linspace(BOUNDS0inf[0], BOUNDS0inf[1], H_res))  # relative weighting of gains/losses
    H_CPT_params['alpha'] =  np.random.choice(np.linspace(BOUNDS0inf[0], BOUNDS0inf[1], H_res))  # prelec parameter
    H_CPT_params['delta'] = np.random.choice(np.linspace(BOUNDS01[0], BOUNDS01[1], H_res))  # convexity gain?
    H_CPT_params['theta'] = 10  # rationality

    # H_CPT_params = {}
    # H_res = 1000
    # # Utility weighting ---------------------
    # H_CPT_params['alpha'] = np.random.choice(np.linspace(.1, .9, H_res))
    # H_CPT_params['beta'] = np.random.choice(np.linspace(.1, .9, H_res))
    # H_CPT_params['lam'] = np.random.choice(np.linspace(.1, 10, H_res))
    # # Probability weighting ------------
    # H_CPT_params['gamma'] = np.random.choice(np.linspace(.1, .9, H_res))
    # H_CPT_params['delta'] = np.random.choice(np.linspace(.1, .9, H_res))
    # # Misc -----------------------------
    # H_CPT_params['theta'] = 10  # np.random.choice(np.linspace(5, 10, H_res))
    # H_CPT_params['b'] = None  # np.random.choice(np.linspace(.1, 10, H_res))
    return H_CPT_params
def get_finite_CPT_models():
    #################################
    # INIT ESTIMATED AGENT ##########
    H_CPT_params = {}
    H_res = 1000
    # Utility weighting ---------------------
    H_CPT_params['alpha'] = np.linspace(.1, .9, H_res)
    H_CPT_params['beta'] = np.linspace(.1, .9, H_res)
    H_CPT_params['lam'] = np.linspace(.1, 10, H_res)
    # Probability weighting ------------
    H_CPT_params['gamma'] = np.linspace(.1, .9, H_res)
    H_CPT_params['delta'] = np.linspace(.1, .9, H_res)
    # Misc -----------------------------
    H_CPT_params['theta'] = 10  # np.random.choice(np.linspace(5, 10, H_res))
    H_CPT_params['b'] = None  # np.random.choice(np.linspace(.1, 10, H_res))
    return H_CPT_params


if __name__ == "__main__":
    main()
