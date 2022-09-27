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

def logit_test(params,ai_obs,u,mu=1.0,pgain = 0.5):
    """get prob of accepting gamble
    Estimate [lam,delta,mu]
    mu: how sensitive people are to diff in subjective utility
    !!!!!!! NEED TO INVERT ACTION INDEX FOR GAMBLE VS CERT !!!!

    """

    #
    # Eu_gamble = np.arange(100)
    # Eu_cert = 0
    # E0 = 5
    # res = 20
    #
    # b = 0
    # lam = 2
    # delta = 0.85
    # gamma = 1
    # alpha = 1
    # theta = 1
    # params = [lam,delta]
    #
    # colors = ['r','k','b']
    # fig, ax = plt.subplots(2, 1)
    # for i,mu in enumerate([.5,1,3]):
    #     P_ACCEPT = []
    #     U_DIFF = []
    #     for E in Eu_gamble:
    #         u_cert = np.random.choice(np.linspace(-E0,E0,res))
    #         u_gain = np.random.choice(np.linspace(-E0, E0, res))
    #         u_loss = np.random.choice(np.linspace(-E0, E0, res))
    #         p_accept,u_diff = logit_test(params,u=[u_cert,-u_loss,u_gain],mu=mu)
    #         P_ACCEPT.append(p_accept)
    #         U_DIFF.append(u_diff)
    #     ax[0].scatter(U_DIFF,P_ACCEPT,label=f'$mu={mu}$',color=colors[i])
    # ax[0].legend()
    # ax[0].set_xlabel("Difference in Subjective Utility \n $E\{u(gamble)\}-E\{u(cert)\}}$")
    # ax[0].set_ylabel("Prob of Accepting \nGamble p(gamble)$")
    # plt.show()











    # lam,delta=params
    # icert,iloss,igain = 0,1,2
    # u_reject = u[icert]
    # u_accept = pgain * np.power(u[igain],delta) + (1-pgain) * (-1*lam*np.power(np.abs(u[iloss]),delta))
    # u_diff = u_accept - u_reject # difference in subjective utility of the two choices
    # p_accept = 1 / (1 + np.exp(-mu * u_diff))
    # return p_accept,u_diff
    # return p_accept,u_diff

    lam,delta=params
    icert,iloss,igain = 0,1,2
    u_reject = u[icert]
    u_accept = pgain * np.power(u[igain],delta) + (1-pgain) * (-1*lam*np.power(np.abs(u[iloss]),delta))
    u_diff = u_accept - u_reject # difference in subjective utility of the two choices


    pi_accept = 1 / (1 + np.exp(-mu * u_diff))


    # y = {1 if choose gamble and 0 if choose cert}
    yi =  ai_obs
    # trial_liklihood = np.dot(np.power(pi_accept,yi) * np.power(1-pi_accept,1-yi))
    negLL = - np.sum(yi*np.log(pi_accept)+(1-yi)(1-np.log(pi_accept)))

    return negLL
def main():

    # Define inference on CPT model =============
    fixed = {}
    fixed['b'] = 0.0
    fixed['gamma'] = 1.0
    # fixed['lam'] = 1.0
    fixed['alpha'] = 1.0
    fixed['delta'] = 1.0
    fixed['theta'] = 1.0

    idx,guess,bounds,cons,CPT_def,fixed = get_CPT_info(fixed)

    # Define H's CPT Model (Ground Truth) =======
    CPT_GT = copy.copy(CPT_def)
    CPT_GT['gamma'] = 0.8  # diminishing return gain
    CPT_GT['lam'] = 8.0  # loss aversion
    for i,key in enumerate(CPT_GT.keys()):
        if not np.isnan(fixed[i]): CPT_GT[key] = fixed[i]

    # Create Samples (observations) from H's CPT model on test cases
    R, T = generate_test_cases()
    # opt_choice, opt_pref = simulate(R, T, CPT_def) # null transformation therefore opt
    # cpt_choice, cpt_pref = simulate(R,T,CPT_GT)

    ai_obs = sample_model(CPT_GT,R, T)
    MLE = MLE_Handler(sample_model,R,T,bounds,cons,fixed = fixed)
    MLE.run(ai_obs,ground_truth=CPT_GT)
    # Define assumtions assumed to be fixed ==================

    # params = np.zeros(6)
    # xtest = np.linspace(0,1,100)
    # P_param = []
    # for i in range(len(xtest)):
    #     params[idx['gamma']] = xtest[i]
    #     negLL = MLE.objective(params,R,T,ai_obs)
    #     P_param.append(negLL)
    # plt.plot(xtest,P_param)
    # plt.show()


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
    eps = 0.0001
    bounds[idx['b']] = (-5.0, 5.0)     #if 'b'       not in fixed else [fixed['b'],fixed['b']]  # reference point
    bounds[idx['gamma']] = (0.0+eps, 1.0+eps)    #if 'gamma'   not in fixed else [fixed['gamma'],fixed['gamma']]  # diminishing return gain
    bounds[idx['lam']] = (1.0, 9.0)    #if 'lam'     not in fixed else [fixed['lam'],fixed['lam']] # loss aversion
    bounds[idx['alpha']] = (0.0+eps, 10)   #if 'alpha'   not in fixed else [fixed['alpha'],fixed['alpha']]  # prelec parameter
    bounds[idx['delta']] = (0.0+eps, 1.0+eps)  #if 'delta'   not in fixed else [fixed['delta'],fixed['delta']]  # Probability weight
    bounds[idx['theta']] = (1.0, 10.0) #if 'theta'   not in fixed else [fixed['theta'],fixed['theta']]  # rationality

    cons = list(np.zeros(len(fixed.keys())))
    icons = 0
    # null_cons = lambda params: 0#params[1] - fixed[]

    if 'b' in fixed:      cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['b']] - fixed['b'])         },icons +1
    if 'gamma' in fixed:  cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['gamma']] - fixed['gamma']) },icons +1
    if 'lam' in fixed:    cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['lam']] - fixed['lam'])     },icons +1
    if 'alpha' in fixed:  cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['alpha']] - fixed['alpha']) },icons +1
    if 'delta' in fixed:  cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['delta']] - fixed['delta']) },icons +1
    if 'theta' in fixed:  cons[icons],icons   = {'type':'eq','fun': (lambda params: params[idx['theta']] - fixed['theta']) },icons +1
    # for i, bnd in enumerate(bounds):
    #     cons.append({'type': 'ineq', 'fun': (lambda params: -1 * (params[i] - bnd[0]))})
    #     cons.append({'type': 'ineq', 'fun': (lambda params: params[i] - bnd[1])})

    return idx,guess,bounds,cons,CPT_def,is_fixed

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

def sample_model(CPT_params,Rn,T, get_pd=False,get_Eu=False):
    nA = T.shape[1]
    nO = np.shape(Rn)[0]
    A = np.arange(nA)
    if isinstance(CPT_params,np.ndarray):
        CPT_params = CPT_arr2dict(CPT_params)

    CPT = CPT_Handler(**CPT_params)

    Eu_perc =  np.zeros([nO, nA])
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
        E_cpt =np.array([np.sum(R_perc * T_perc[0, ai, :]) for ai in A])

        # print(E_cpt-E_opt)
        Eu_perc[obs] = np.copy(E_cpt)
        cpt_pref[obs] = noisy_rational(E_cpt, rationality=CPT.lam)
        # try:
        cpt_choice[obs] = np.random.choice([0,1],p=cpt_pref[obs])
        # except:
        #     pass

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
        self._def_guess = [np.mean(self.bnds[ip]) if np.isnan(fixed[ip]) else fixed[ip] for ip in range(self.nparams)]
        self.names = ['b','gamma','lam','alpha','delta','theta']


        # Define Solver ===============
        self.solver = None
        # self.solver = 'L-BFGS-B'
        # self.solver = 'BFGS'
        # self.solver = 'SLSQP'
        # self.solver = 'Nelder-Mead'

    def objective(self, parameters, R,T, ai_obs):
        """
        https://www.thegreatstatsby.com/posts/2021-03-08-ml-prospect/
        """
        # try:
            #  difference in subjective utility between the two options: u(accept)−u(reject):

        ireject, iaccept = 0,1
        icert,icert0, iloss, igain = 0, 1, 2, 3
        ai_exp, pai,Eui = self.mdl(parameters, R, T, get_pd=True,get_Eu=True)

        # u_reject = Eui[:,ireject]
        # u_accept = Eui[:,iaccept]
        # u_diff = u_accept - u_reject  # difference in subjective utility of the two choices
        # mu = parameters[-1] # mu is rationality (sensitivitgy between choices)
        # pi_accept = 1 / (1 + np.exp(-mu * u_diff))
        # # ai_BR = np.argmax(Eui,axis=1)
        pi_accept = pai[:,iaccept]
        yi = ai_obs  # y = {1 if choose gamble; 0 if choose cert}



        LLi = yi * np.log(pi_accept) + (1 - yi)*(1 - np.log(pi_accept))
        # LLnorm = np.linalg.norm(LLi)
        # negLL = - np.sum(LLi / LLnorm)
        negLL = - np.sum(LLi)
        #negLL = - np.sum(stats.norm.logcdf(ai_obs - ai_exp, loc=0, scale=parameters[-1]))
        return negLL

        # Liklihood
        # L = np.sum(np.power(pi_accept,yi) * np.power(1 - pi_accept, 1 - yi))
        # return L


    def run(self, ai_obs, guess=None, ground_truth=None):
        guess = self._def_guess if guess is None else guess
        print(f'Running MLE:')
        print(f'\t| N samples = {ai_obs.size}')
        print(f'\t| Fixed:')
        for i, key in enumerate(self.names): print(f'\t\t| {key}  = \t[{np.round(self.fixed[i],2)}]')
        print(f'\t| Bounds:')
        for i, key in enumerate(self.names): print(f'\t\t| {key}  = \t[{np.round(self.bnds[i],2)}]')


        # print(f'Optim')
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
        # if ground_truth is not None:  print(
        #     f'\t| %Error    = {np.round(100 * ( GT_array - self.x_star) / (GT_array+0.0001), 2)}')
        return self.x_star


if __name__ == "__main__":
    main()
