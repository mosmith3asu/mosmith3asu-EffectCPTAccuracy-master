import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from math import pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import random
from random import randrange
from agents.policies import noisy_rational,CPT_Handler
from functools import partial
np.random.seed(0)
random.seed(0)
def main():

    # Define inference on CPT model =============
    fixed = {}
    # fixed['b'] = 0.0
    # fixed['gamma'] = 1.0
    # fixed['lam'] = 1.0
    # fixed['alpha'] = 1.0
    # fixed['delta'] = 1.0
    # fixed['theta'] = 1.0

    idx,guess,bounds,cons,CPT_def,fixed = get_CPT_info(fixed)

    # Define H's CPT Model (Ground Truth) =======
    CPT_GT = copy.copy(CPT_def)
    CPT_GT['gamma'] = 0.8                               # diminishing return gain
    CPT_GT['lam'] = 2.0                                 # loss aversion
    CPT_GT['alpha'] = 1.0  # loss aversion
    CPT_GT['delta'] = 0.6
    for i,key in enumerate(CPT_GT.keys()):
        if not np.isnan(fixed[i]): CPT_GT[key] = fixed[i]

    # Create Samples (observations) from H's CPT model on test cases
    R, T = generate_test_cases()
    ai_obs = sample_model(CPT_GT,R, T)
    MLE = MLE_Handler(sample_model,R,T,bounds,cons,fixed = fixed)
    xstar = MLE.run(ai_obs,ground_truth=CPT_GT)


    # Create PDF ================
    # name = 'gamma'
    # params = np.array([CPT_def[key] for key in CPT_def])
    # xtest = np.linspace(0.2, 2, 100)

    name = 'lam'
    params = np.array([CPT_def[key] for key in CPT_def])
    # params = np.copy(xstar)
    xtest = np.linspace(0.2,5,100)
    Li = []
    LLi = []
    obj = []
    for i in range(len(xtest)):
        params[idx[name]] = xtest[i]
        Li.append(MLE.liklihood(params, R, T, ai_obs))
        LLi.append(MLE.log_liklihood(params, R, T, ai_obs))
        obj.append(MLE.objective(params, R, T, ai_obs))
    fig,axs = plt.subplots(3,1)
    axs[0].set_title(f'Liklihood of {name} | $x^*$')
    axs[0].plot(xtest,Li)
    axs[1].set_title(f'Log Liklihood of {name} | $x^*$')
    axs[1].plot(xtest, LLi)
    axs[2].set_title(f'Objective Fun of {name} | $x^*$')
    axs[2].plot(xtest, obj)
    plt.show()


def CPT_arr2dict(A):
    idx = {}
    idx['b'] = 0
    idx['gamma'] = 1  # diminishing return gain
    idx['lam'] = 2  # loss aversion
    idx['alpha'] = 3  # prelec parameter
    idx['delta'] = 4  # Probability weight
    idx['theta'] = 5  # rationality

    CPT_params = {}
    for key in idx:  CPT_params[key] = A[idx[key]]
    return CPT_params
def get_CPT_info(fixed):
    CPT_def = {}
    CPT_def['b'] = 0.0  # reference point
    CPT_def['gamma'] = 1.0  # diminishing return gain
    CPT_def['lam'] = 1.0  # loss aversion
    CPT_def['alpha'] = 1.0  # prelec parameter
    CPT_def['delta'] = 1.0  # Probability weight
    CPT_def['theta'] = 1.0  # rationality

    is_fixed = [ fixed[key] if key in fixed.keys() else np.nan  for key in CPT_def.keys()]


    idx = {}
    idx['b'] = 0
    idx['gamma'] = 1  # diminishing return gain
    idx['lam'] = 2  # loss aversion
    idx['alpha'] = 3  # prelec parameter
    idx['delta'] = 4  # Probability weight
    idx['theta'] = 5  # rationality

    n_params = len(idx.keys())
    guess = np.zeros(n_params)
    guess[idx['b']] = 0.0  # reference point
    guess[idx['gamma']] = 1.0  # diminishing return gain
    guess[idx['lam']] = 1.0  # loss aversion
    guess[idx['alpha']] = 1.0  # prelec parameter
    guess[idx['delta']] = 1.0  # Probability weight
    guess[idx['theta']] = 1.0  # rationality

    bounds = list(np.zeros(n_params))
    eps = 0.01
    # bounds[idx['b']] = (-5.0, 5.0)
    # bounds[idx['gamma']] = (0.0+eps, 1.0+eps)
    # bounds[idx['lam']] = (1.0, 9.0)
    # bounds[idx['alpha']] = (0.0+eps, 10)
    # bounds[idx['delta']] = (0.0+eps, 1.0+eps)
    # bounds[idx['theta']] = (1.0, 10.0)


    # bounds[idx['b']] = (-5.0, 5.0)
    # bounds[idx['gamma']] = (0.2, 2.0)
    # bounds[idx['lam']] = (0.2, 5.0)
    # bounds[idx['alpha']] = (0.2, 3.0)
    # bounds[idx['delta']] = (0.2, 0.3)
    # bounds[idx['theta']] = (1.0, 10.0)

    bounds[idx['b']]     = (0,0)
    bounds[idx['gamma']] = (0+eps,2+eps)
    bounds[idx['lam']]   = (0+eps,2+eps)
    bounds[idx['alpha']] = (0+eps,2+eps)
    bounds[idx['delta']] = (0+eps,2+eps)
    bounds[idx['theta']] = (1,1)

    cons = list(np.zeros(len(fixed.keys())))
    icons = 0
    if 'b' in fixed:      cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['b']] - fixed['b'])         },icons +1
    if 'gamma' in fixed:  cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['gamma']] - fixed['gamma']) },icons +1
    if 'lam' in fixed:    cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['lam']] - fixed['lam'])     },icons +1
    if 'alpha' in fixed:  cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['alpha']] - fixed['alpha']) },icons +1
    if 'delta' in fixed:  cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['delta']] - fixed['delta']) },icons +1
    if 'theta' in fixed:  cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['theta']] - fixed['theta']) },icons +1

    return idx,guess,bounds,cons,CPT_def,is_fixed
def generate_test_cases(debug=False):
    # np.random.seed(0)
    nS = 4
    nA = 2
    nO = 100
    icert,igamble = 0,1
    r_penalty = -3
    p_penalty = 0.5
    range_cert = [0,18]
    range_uncert = [0,18]
    T = np.zeros([1,nA,nS])
    T[0, icert, :] = [1, 0, 0, 0]
    T[0, igamble, :] = [0,0,1-p_penalty,p_penalty]
    Rn = np.zeros([nO, nS])
    # En = np.zeros([nO, nA])
    for obs in range(nO):
        E_cert = randrange(range_cert[0],range_cert[1])
        E_uncert = randrange(range_uncert[0],range_uncert[1])
        Rn[obs] = [E_cert, 0, E_uncert - r_penalty, E_uncert]
        if debug: Rn[obs] = [E_cert, 0, E_cert-r_penalty, E_cert+r_penalty]

    # Eai = np.zeros([nO,2])
    # Eai[:,icert] = np.sum(T[0, icert, :]*Rn,axis=1)
    # Eai[:, igamble] = np.sum(T[0, igamble, :] * Rn,axis=1)

    return Rn,T

def CPT_transformation(params,R,T,choose=False):
    b, gamma, lam, alpha, delta, theta = params

    # Utility Transformation
    igain = np.where(R - b >= 0)
    iloss = np.where(R - b < 0)
    wr = np.zeros(np.shape(R))
    wr[igain] = np.power(R[igain],gamma)
    wr[iloss] = -lam*np.power(-R[iloss],gamma)

    # Probability transformation
    wp = np.zeros(T.shape)
    wp[np.where(T>0)] = np.exp(-alpha*np.power(-np.log(T[np.where(T>0)]),delta))

    # Percieved Expected value and choice
    Eai= np.array([np.sum(wp[0, ia, :] * wr) for ia in range(T.shape[1])])

    u_diff = Eai[1] - Eai[0]
    p_gamble = 1 / (1 + np.exp(-theta * u_diff))
    pda = np.array([1 - p_gamble, p_gamble])
    if choose:
        a_choice = np.random.choice([0,1],p=pda)
        return a_choice
    return Eai,pda




def sample_model(CPT_params,Rn,T, get_pd=False,get_Eu=False):
    icert, igamble = 0, 1
    nA = T.shape[1]
    nO = np.shape(Rn)[0]
    A = np.arange(nA)
    # if isinstance(CPT_params,np.ndarray): CPT_params = CPT_arr2dict(CPT_params)
    if isinstance(CPT_params,dict): CPT_params = np.array([CPT_params[key] for key in CPT_params.keys()])

    # CPT = CPT_Handler(**CPT_params)
    Eu_perc =  np.zeros([nO, nA])
    opt_pref = np.zeros([nO, nA])
    cpt_pref = np.zeros([nO, nA])
    cpt_choice = np.zeros(nO,dtype='int8')
    for obs in range(nO):
        # E_opt =np.array([np.sum(Rn[obs] * T[0, ai, :]) for ai in A])
        # opt_pref[obs] = noisy_rational(E_opt, rationality=1)
        #
        # T_perc = np.zeros(np.shape(T))
        # T_perc[0, icert, :] = T[0, icert, :] #CPT.prob_weight(T[0, icert, :])
        # T_perc[0, igamble, :] = CPT.prob_weight(T[0, igamble, :])
        # R_perc = CPT.utility_weight(Rn[obs])
        #
        # Eu_perc[obs, icert] = np.sum(T_perc[0, icert, :] * R_perc)
        # Eu_perc[obs, igamble] = np.sum(T_perc[0, igamble, :] * R_perc)
        # cpt_pref[obs] = noisy_rational(Eu_perc[obs,:], rationality=CPT.lam)
        # cpt_choice[obs] = np.random.choice([0,1],p=cpt_pref[obs])

        Eai_perc,pda_perc = CPT_transformation(CPT_params,Rn[obs],T)
        Eu_perc[obs,:] = Eai_perc
        cpt_pref[obs] = pda_perc
        cpt_choice[obs] = CPT_transformation(CPT_params,Rn[obs],T,choose=True)

    if np.all([get_pd, get_Eu]): return cpt_choice,cpt_pref,Eu_perc
    elif get_pd: return cpt_choice,cpt_pref
    elif get_Eu: return cpt_choice, Eu_perc
    else: return cpt_choice


class MLE_Handler():
    def __init__(self,mdl, R,T,bnds,cons,fixed=None):
        # Define Aux Params ==============
        self.mdl = mdl
        self.R,self.T = R,T
        self.bnds = bnds
        self.cons = cons
        self.fixed = fixed
        self.nparams = len(bnds)
        # self._def_guess = [np.mean(self.bnds[ip]) for ip in range(self.nparams)]
        try: self._def_guess = [np.mean(self.bnds[ip]) if np.isnan(fixed[ip]) else fixed[ip] for ip in range(self.nparams)]
        except: self._def_guess = [0,1,1,1,1,1]
        self.names = ['b','gamma','lam','alpha','delta','theta']


        # Define Solver ===============
        self.solver = None
        self.solver = 'L-BFGS-B'
        # self.solver = 'BFGS'
        # self.solver = 'SLSQP'
        # self.solver = 'Nelder-Mead'


    def liklihood(self, parameters, R,T, ai_obs):
        ai_exp, pai, Eui = self.mdl(parameters, R, T, get_pd=True, get_Eu=True)
        Li = pai[:, ai_obs]#/np.linalg.norm( pai[:, ai_obs])
        return np.sum(Li)#np.product(Li)
    def log_liklihood(self, parameters, R,T, ai_obs):
        ai_exp, pai, Eui = self.mdl(parameters, R, T, get_pd=True, get_Eu=True)

        # Manually ====================
        icert,igamble = 0,1
        mu = parameters[-1]
        u_reject = Eui[:,icert]
        u_accept = Eui[:,igamble]
        u_diff = u_accept - u_reject
        p_accept = 1/(1+np.exp(-mu*u_diff))
        LLi = (ai_obs)*p_accept + (1-ai_obs)*(1-p_accept)

        # Intuitively ====================
        # LLi = np.log(pai[:, ai_obs])
        return np.sum(LLi)


    def objective(self, parameters, R,T, ai_obs):
        """ https://www.thegreatstatsby.com/posts/2021-03-08-ml-prospect/ """
        return -1 * self.log_liklihood(parameters, R, T, ai_obs)



    def run(self, ai_obs, guess=None, ground_truth=None,verbose=False):
        guess = self._def_guess if guess is None else guess
        print(f'Running MLE:')
        print(f'\t| N samples = {ai_obs.size}')
        print(f'\t| Fixed:')
        for i, key in enumerate(self.names): print(f'\t\t| {key}  = \t[{np.round(self.fixed[i],2)}]')
        print(f'\t| Bounds:')
        for i, key in enumerate(self.names):
            try: print(f'\t\t| {key}  = \t[{np.round(self.bnds[i],2)}]')
            except: print(f'\t\t| {key}  = \t[{self.bnds[i]}]')

        result = minimize(self.objective, guess,
                          args=(self.R,self.T,np.array(ai_obs),),
                          # tol=1e-10,
                          constraints=self.cons,
                          bounds=self.bnds,
                          method=self.solver,
                          options={'disp':verbose}
                          )
        self.x_star = result['x']


        print(f'\t| Guess     = {np.round(guess, 2)}')
        if ground_truth is not None:
            GT_array = np.array([ground_truth[key] for key in ground_truth])
            print(f'\t| Truth     = {np.round(GT_array, 2)}')

        print(f'\t| Recovered = {np.round(self.x_star, 2)}')
        # if ground_truth is not None:  print(
        #     f'\t| %Error    = {np.round(100 * ( GT_array - self.x_star) / (GT_array+0.0001), 2)}')
        return self.x_star


if __name__ == "__main__":
    main()
