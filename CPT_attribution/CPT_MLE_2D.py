import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from math import pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from random import randrange
from agents.policies import noisy_rational,CPT_Handler
from functools import partial
# np.random.seed(0)


def main():

    # Define inference on CPT model =============
    fixed = {}
    fixed['b'] = 0.0
    fixed['gamma'] = 1.0
    # fixed['lam'] = 1.0
    fixed['alpha'] = 1.0
    fixed['delta'] = 1.0
    fixed['theta'] = 1.0
    idx,guess,bounds,cons,CPT_def = get_CPT_info(fixed)

    # Define H's CPT Model (Ground Truth) =======
    CPT_GT = copy.copy(CPT_def)
    # CPT_GT['gamma'] = 0.5  # diminishing return gain
    CPT_GT['lam'] = 8.0  # loss aversion

    # Create Samples (observations) from H's CPT model on test cases
    R, T = generate_test_cases()
    # opt_choice, opt_pref = simulate(R, T, CPT_def) # null transformation therefore opt
    # cpt_choice, cpt_pref = simulate(R,T,CPT_GT)

    ai_obs = sample_model(CPT_GT,R, T)
    MLE = MLE_Handler(sample_model,R,T,bounds,cons)
    MLE.run(ai_obs,ground_truth=CPT_GT)
    # Define assumtions assumed to be fixed ==================



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
    bounds[idx['b']] = (-5.0, 5.0)     if 'b'       not in fixed else [fixed['b'],fixed['b']]  # reference point
    bounds[idx['gamma']] = (0, 1.0)    if 'gamma'   not in fixed else [fixed['gamma'],fixed['gamma']]  # diminishing return gain
    bounds[idx['lam']] = (1.0, 9.0)    if 'lam'     not in fixed else [fixed['lam'],fixed['lam']] # loss aversion
    bounds[idx['alpha']] = (0.0, 10)   if 'alpha'   not in fixed else [fixed['alpha'],fixed['alpha']]  # prelec parameter
    bounds[idx['delta']] = (0.0, 1.0)  if 'delta'   not in fixed else [fixed['delta'],fixed['delta']]  # Probability weight
    bounds[idx['theta']] = (1.0, 10.0) if 'theta'   not in fixed else [fixed['theta'],fixed['theta']]  # rationality

    cons = list(np.zeros(n_params))
    null_cons = lambda params: params[1]
    cons[idx['b']]      = {'type':'eq','fun': (lambda params: params[idx['b']] - fixed['b']) if 'b' in fixed else null_cons }
    cons[idx['gamma']]  = {'type':'eq','fun': (lambda params: params[idx['gamma']] - fixed['gamma']) if 'gamma' in fixed else null_cons}
    cons[idx['lam']]    = {'type':'eq','fun': (lambda params: params[idx['lam']] - fixed['lam']) if 'lam' in fixed else null_cons}
    cons[idx['alpha']]  = {'type':'eq','fun': (lambda params: params[idx['alpha']] - fixed['alpha']) if 'alpha' in fixed else null_cons}
    cons[idx['delta']]  = {'type':'eq','fun': (lambda params: params[idx['delta']] - fixed['delta']) if 'delta' in fixed else null_cons}
    cons[idx['theta']]  = {'type':'eq','fun': (lambda params: params[idx['theta']] - fixed['theta']) if 'theta' in fixed else null_cons}

    return idx,guess,bounds,cons,CPT_def

def generate_test_cases():
    nS = 4
    nA = 2
    nO = 100
    r_penalty = -3
    p_penalty = 0.5
    range_cert = [0,18]
    range_uncert = [0,18]
    T = np.zeros([1,nA,nS])
    T[0, 0, :] = [1, 0, 0, 0]
    T[0, 1, :] = [0,0,1-p_penalty,p_penalty]
    Rn = np.zeros([nO, nS])
    En = np.zeros([nO, nA])
    for obs in range(nO):
        E_cert = randrange(range_cert[0],range_cert[1])
        E_uncert = randrange(range_uncert[0],range_uncert[1])
        Rn[obs] = [E_cert, 0, E_uncert-r_penalty, E_uncert+r_penalty]
    return Rn,T

def sample_model(CPT_params,Rn,T,get_pd=False):
    nA = T.shape[1]
    nO = np.shape(Rn)[0]
    A = np.arange(nA)
    if isinstance(CPT_params,np.ndarray):
        CPT_params = CPT_arr2dict(CPT_params)

    CPT = CPT_Handler(**CPT_params)

    opt_pref = np.zeros([nO, nA])
    cpt_pref = np.zeros([nO, nA])
    cpt_choice = np.zeros(nO,dtype='int8')
    for obs in range(nO):
        E_opt =np.array([np.sum(Rn[obs] * T[0, ai, :]) for ai in A])
        opt_pref[obs] = noisy_rational(E_opt, rationality=1)

        T_perc = np.zeros(np.shape(T))
        T_perc[0, 0, :] = CPT.prob_weight(T[0, 0, :])
        T_perc[0, 1, :] = CPT.prob_weight(T[0, 1, :])
        R_perc = CPT.utility_weight(Rn[obs])
        E_cpt =np.array( [np.sum(R_perc * T_perc[0, ai, :]) for ai in A])

        # print(E_cpt-E_opt)
        cpt_pref[obs] = noisy_rational(E_cpt, rationality=CPT.lam)

        cpt_choice[obs] = np.random.choice([0,1],p=cpt_pref[obs])
    if get_pd: return cpt_choice,cpt_pref
    return cpt_choice






class MLE_Handler():
    def __init__(self,mdl, R,T,bnds,cons):
        # Define Aux Params ==============
        self.mdl = mdl
        self.R,self.T = R,T
        self.bnds = bnds
        self.cons = cons
        self.nparams = len(bnds)
        self._def_guess = [np.mean(self.bnds[ip]) for ip in range(self.nparams)]

        # Define Solver ===============
        self.solver = None
        # self.solver = 'L-BFGS-B'
        # self.solver = 'BFGS'
        # self.solver = 'SLSQP'
        # self.solver = 'Nelder-Mead'

    # def log_liklihood(self, parameters, t, y_obs):
    #     y_exp = self.mdl(parameters)
    #     pdf_L = np.sum(stats.norm.logpdf(y_obs, y_exp, parameters[-1]))
    #     return pdf_L
    #
    # def liklihood(self, parameters, t, y_obs):
    #     y_exp = sample_model(parameters, t, noise=False)
    #     pdf_L = np.sum(stats.norm.pdf(y_obs, y_exp, parameters[-1]))
    #     return pdf_L

    def objective(self, parameters, R,T, ai_obs):

        # pmf used for discrete prob distributiosn
        """ last parameter is stdv residual between models"""
        # y_exp = sample_model(parameters, t, noise=False)
        # neg_LL = - np.sum(stats.norm.logpdf(y_obs, y_exp, parameters[-1]))
        # return neg_LL
        ai_exp,pai = self.mdl(parameters,R,T,get_pd=True)
        neg_LL = -np.sum(np.log(pai[ai_obs]))
        # neg_LL = -np.sum(stats.norm.(pai[ai_obs]))
        return neg_LL





    def run(self, ai_obs, guess=None, ground_truth=None):
        guess = self._def_guess if guess is None else guess
        print(f'Running MLE:')
        print(f'\t| N samples = {ai_obs.size}')
        print(f'\t| Guess     = {np.round(guess, 2)}')
        if ground_truth is not None:
            GT_array = np.array([ground_truth[key] for key in ground_truth])
            print(f'\t| Truth     = {np.round(GT_array, 2)}')

        result = minimize(self.objective, guess,
                          args=(self.R,self.T,np.array(ai_obs),),
                          tol=1e-10,
                          constraints=self.cons,
                          bounds=self.bnds,
                          method=self.solver,
                          # options={'disp':True}
                          )
        self.x_star = result['x']

        print(f'\t| Recovered = {np.round(self.x_star, 2)}')
        if ground_truth is not None:  print(
            f'\t| %Error    = {np.round(100 * ( GT_array - self.x_star) / (GT_array+0.0001), 2)}')
        return self.x_star


if __name__ == "__main__":
    main()
