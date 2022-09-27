# import numpy as np
# import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from agents.policies import noisy_rational,CPT_Handler
import itertools
import tqdm
from scipy import stats
def main():

    # bnds = []
    # bnds.append((-1, 1))  # b
    # bnds.append((0,1)) # gamma
    # bnds.append((0.2,5)) # lam
    # bnds.append((0.2,5)) # alpha
    # bnds.append((0.2,1)) # delta
    # bnds.append((1,5)) # theta

    res = 5
    param_range = []
    param_range.append(np.array([0]))  # b
    param_range.append(np.linspace(0.01,1,res))  # gamma
    param_range.append(np.linspace(0.01,5,res))  # lam     param_range.append([1])  # lam
    param_range.append(np.linspace(0.01,5,res))  # alpha
    param_range.append(np.linspace(0.01,1,res))  # delta
    param_range.append(np.array([1]))  # theta # param_range.append(np.linspace(0.01,1,res)) # theta

    # res = 5
    # param_range = [np.linspace(bnd[0],bnd[1],res) for bnd in bnds]
    Models = np.array(list(itertools.product(*param_range)))
    nM = Models.shape[0]

    # Generate Observations under known model
    params0 = [0.0,1.0,1.0,1.0,1.0,1.0]
    Rn, T = generate_test_cases()
    ai_obs = sample_model(params0, Rn, T)

    nLL_iM = np.zeros([nM,1])
    for iM in tqdm.tqdm(range(nM)):
        model = Models[iM,:]
        nLL_iM[iM] = negLL(model, Rn, T, ai_obs)
    iMstar = np.argmax(nLL_iM)
    print(f'truth = {params0} ')
    print(f'MLE   = {Models[iMstar]}')
    print(f'nLL*   = {nLL_iM[iMstar]}')


def generate_test_cases(debug=False):
    np.random.seed(0)
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

def CPT_arr2dict(A):
    idx = {}
    idx['b'] = 0
    idx['gamma'] = 1  # diminishing return gain
    idx['lam'] = 2  # loss aversion
    idx['alpha'] = 3  # prelec parameter
    idx['delta'] = 4  # Probability weight
    idx['theta'] = 5  # rationality
    ioff=0

    CPT_params = {}
    if len(A) == 5:
        CPT_params['b'] = 0
        ioff = 1
    for key in idx:
        CPT_params[key] = A[idx[key]-ioff]
    return CPT_params
def sample_model(CPT_params,Rn,T, get_pd=False,get_Eu=False):
    icert, igamble = 0, 1
    nA = T.shape[1]
    nO = np.shape(Rn)[0]
    A = np.arange(nA)
    if isinstance(CPT_params, list): CPT_params = np.array(CPT_params)
    if isinstance(CPT_params,np.ndarray): CPT_params = CPT_arr2dict(CPT_params)
    if len(CPT_params.keys()) ==5: CPT_params['b'] = 0

    CPT = CPT_Handler(**CPT_params)
    Eu_perc =  np.zeros([nO, nA])
    opt_pref = np.zeros([nO, nA])
    cpt_pref = np.zeros([nO, nA])
    cpt_choice = np.zeros(nO,dtype='int8')
    for obs in range(nO):
        E_opt =np.array([np.sum(Rn[obs] * T[0, ai, :]) for ai in A])
        opt_pref[obs] = noisy_rational(E_opt, rationality=1)

        T_perc = np.zeros(np.shape(T))
        T_perc[0, icert, :] = T[0, icert, :] #CPT.prob_weight(T[0, icert, :])
        T_perc[0, igamble, :] = CPT.prob_weight(T[0, igamble, :])
        R_perc = CPT.utility_weight(Rn[obs])

        Eu_perc[obs, icert] = np.sum(T_perc[0, icert, :] * R_perc)
        Eu_perc[obs, igamble] = np.sum(T_perc[0, igamble, :] * R_perc)
        cpt_pref[obs] = noisy_rational(Eu_perc[obs,:], rationality=CPT.lam)
        cpt_choice[obs] = np.random.choice([0,1],p=cpt_pref[obs])

    if np.all([get_pd, get_Eu]): return cpt_choice,cpt_pref,Eu_perc
    elif get_pd: return cpt_choice,cpt_pref
    elif get_Eu: return cpt_choice, Eu_perc
    else: return cpt_choice


def negLL(parameters, R,T, ai_obs):
    ai_exp, pai, Eui = sample_model(parameters, R, T, get_pd=True, get_Eu=True)
    nLL = -1*np.sum(np.log(pai[:, ai_obs]))  # /np.linalg.norm( pai[:, ai_obs])
    return nLL
if __name__ == "__main__":
    main()
