# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy.optimize import minimize
from scipy import stats
import pandas



def main():
    global param_idx
    #######################################################
    # Define Global settings ##############################
    param_idx = {'gamma': 0, 'lam': 1, 'alpha': 2, 'delta': 3, 'theta': 4}
    fixed_params = {'delta': 1.0, 'theta': 1.0} # dict of param name and value to be fixed

    #######################################################
    # Human and Environment ###############################
    Rn, T = generate_uniform_test_cases(r_diff=5)  # Generate decisions rewards and probability
    params0 = get_human_model(**fixed_params)  # Define ground truth human model

    # Simulate human observations
    ai_obs = np.zeros([len(Rn), 1])
    for iobs, R in enumerate(Rn):
        p_gamble = CPT_model(params0, R)
        ai_obs[iobs] = np.random.choice([0, 1], p=[1 - p_gamble, p_gamble])

    #######################################################
    # Perform MLE #########################################
    guess, bnds, cons = get_opt_params(**fixed_params)  # inputs fix params as constant
    result = minimize(objective, np.array(guess),
                      args=(np.array(ai_obs), Rn,),
                      constraints=cons, bounds=bnds,
                      # method=self.solver, options={'disp':True}
                      )
    xstar = result['x']

    #######################################################
    # Report MLE ##########################################
    col_labels = ['gamma','lam','alpha','delta','theta']
    rowdata = {'Bounds':[(round(bnd[0],2),round(bnd[1],2)) for bnd in bnds],
               'Guess': guess,
               'True': params0,
               'MLE': np.round(xstar,2),
               'err': list((np.array(guess)-np.array(xstar).round(2))) }
    df = pandas.DataFrame.from_dict(rowdata,orient='index',columns=col_labels)
    df = df.round(2)
    print(df)

    # Plot ==============
    labels = ['$\\gamma$\n(diminishing rewards)',   '$\\lambda$\n(loss aversion)',
              '$\\alpha$\n(over/underest. prob.)',  '$\\delta$\n(diminishing prob.)',
              '$\\theta$\n(rationality)']
    x = np.arange(len(labels))  # the label locations



    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(2,2)

    # PARAMETER RECOVERY BARPLOT
    rects1 = ax[0,0].bar(x - width / 2, params0, width, label='$\mathcal{X}^{0}_{CPT}$',color='k')
    rects2 = ax[0,0].bar(x + width / 2, xstar[:len(params0)], width, label='$\mathcal{X}^{*}_{CPT}$',color='r')
    ax[0, 0].set_ylabel('Parameter Value')
    ax[0, 0].set_title('MLE Parameter Recovery')
    ax[0, 0].set_xticks(x, labels)
    ax[0, 0].legend()
    ax[0, 0].bar_label(rects1, padding=3)
    ax[0, 0].bar_label(rects2, padding=3)

    # REWARD TRANSFORMATION
    rT0, pT0 = CPT_transform(params0) # human CPT transform
    rTstar, pTstar = CPT_transform(xstar)
    ax[1, 0].vlines(0, ymin=min(rT0[0]), ymax=max(rT0[0]), color='lightgrey', ls=':')
    ax[1, 0].hlines(0, xmin=min(rTstar[0]), xmax=max(rTstar[0]), color='lightgrey', ls=':')
    ax[1, 0].plot(rT0[0],rT0[1],label='$\mathcal{X}^{0}_{CPT}$')
    ax[1, 0].plot(rTstar[0], rTstar[1], label='$\mathcal{X}^{*}_{CPT}$')
    ax[1, 0].set_title('Reward Transformation')
    ax[1, 0].set_ylabel('Perceived Reward')
    ax[1, 0].set_xlabel('Relative Reward $(r-b)$')
    ax[1, 0].vlines(0, ymin=min(rT0[0]), ymax=max(rT0[0]), color='lightgrey', ls=':')
    ax[1, 0].hlines(0, xmin=min(rTstar[0]), xmax=max(rTstar[0]), color='lightgrey', ls=':')
    ax[1, 0].legend()

    ax[1, 1].plot(pT0[0], pT0[0], label='Optimal', color='lightgrey', ls=':')
    ax[1, 1].plot(pT0[0], pT0[1], label='$\mathcal{X}^{0}_{CPT}$')
    ax[1, 1].plot(pTstar[0], pTstar[1], label='$\mathcal{X}^{*}_{CPT}$')
    ax[1, 1].set_title('Probability Transformation')
    ax[1, 1].set_ylabel('Perceived Probability')
    ax[1, 1].set_xlabel('Probability $(p)$')
    ax[1, 1].legend()
    # PROBABILITY TRANSFORMATION


    # fig.tight_layout()
    plt.show()
def get_human_model(imodel=0,**kwargs):
    global param_idx
    if imodel==0: # get random human model within reasonable bounds
        res = 100
        gamma0 = np.random.choice(np.linspace(0.01,0.99,res))
        lam0   = np.random.choice(np.linspace(0.20,3.00,res))
        alpha_tmp = np.linspace(1.01,5.00,int(res/2)) # need to make |a>1|=|a<1| for fair sampling of inverse behaviors
        alpha0 = np.random.choice(np.append(np.power(alpha_tmp,-1),alpha_tmp))
        delta0 = np.random.choice(np.linspace(0.01,0.99,res))
        theta0 = np.random.choice(np.linspace(0.20,5.00,res))
    elif imodel==1:
        gamma0 = 0.80
        lam0    = 1.52
        alpha0 = 1.89
        delta0 = 0.94
        theta0 = 1.00
    elif imodel==1: # loss averse
        gamma0 = 1.00
        lam0   = 2.00
        alpha0 = 1.00
        delta0 = 1.00
        theta0 = 1.00
    else: # optimal
        gamma0 = 1.00
        lam0   = 1.00
        alpha0 = 1.00
        delta0 = 1.00
        theta0 = 1.00
    params0 = [gamma0, lam0, alpha0, delta0, theta0]

    # Update fixed parameters
    if len(kwargs.keys())>0:
        print(f'\nFixing H Model Params:')
    for key in kwargs:
        print(f'\t|{key} = {kwargs[key]}')
        params0[param_idx[key]] =  kwargs[key]
    # if 'gamma' == key: theta0 = kwargs['theta']
    # if 'lam'   in kwargs.keys(): theta0 = kwargs['lam']
    # if 'alpha' in kwargs.keys(): theta0 = kwargs['alpha']
    # if 'delta' in kwargs.keys(): theta0 = kwargs['delta']
    # if 'theta' in kwargs.keys(): theta0 = kwargs['theta']
    return params0
def get_opt_params(**kwargs):
    global param_idx
    gamma1, gamma_bnds = 0.87, (0.01,0.99)      # gamma1, gamma_bnds = 0.87, (0.01,0.99)
    lam1,   lam_bnds   = 1.27, (0.20,5.00)      # lam1,   lam_bnds   = 1.27, (1.00,5.00)
    alpha1, alpha_bnds = 2.10, (0.20,5.00)      # alpha1, alpha_bnds = 2.10, (1.00,5.00)
    delta1, delta_bnds = 0.80, (0.01,0.99)      # delta1, delta_bnds = 0.80, (0.01,0.99)
    theta1, theta_bnds = 1.00, (0.50,5.00)      # theta1, theta_bnds = 1.00, (1.00,5.00)
    guess = [gamma1,lam1,alpha1,delta1,theta1]
    bnds = [gamma_bnds, lam_bnds, alpha_bnds, delta_bnds, theta_bnds]
    cons = {}  # {'type': 'ineq', 'fun': lambda params: params[-1]}  # constrain reward sensitivity (rationality) >0

    # Update fixed parameters
    eps = 1e-8
    if len(kwargs.keys()) > 0:
        print(f'Fixing MLE Params:')
    for key in kwargs:
        print(f'\t|{key} = {kwargs[key]}')
        bnds[param_idx[key]] = (kwargs[key] - eps, kwargs[key] + eps)
        guess[param_idx[key]] = kwargs[key]
        # if   key == 'gamma': bnds[0],guess[0] = (kwargs[key] - eps, kwargs[key] + eps),kwargs[key]
        # elif key == 'lam':   bnds[1],guess[1] = (kwargs[key] - eps, kwargs[key] + eps),kwargs[key]
        # elif key == 'alpha': bnds[2], guess[2] = (kwargs[key] - eps, kwargs[key] + eps),kwargs[key]
        # elif key == 'delta': bnds[3],guess[3] = (kwargs[key] - eps, kwargs[key] + eps),kwargs[key]
        # elif key == 'theta': bnds[4], guess[4] = (kwargs[key] - eps, kwargs[key] + eps),kwargs[key]
        #else: raise Exception('Fixed CPT parameter not found in "get_opt_params()"')

    return guess,bnds,cons


def CPT_transform(params):
    gamma, lam, alpha, delta, theta = params
    prelec = lambda p, a, d: np.exp(-a * np.power(-np.log(p), d))
    rtrans_gain = lambda r, g: np.power(r, g)
    rtrans_loss = lambda r, l, g: -l * np.power(-r, g)

    res = 100
    ptest = np.linspace(0.01, 0.99, res)
    rtest = np.linspace(-10, 10, res)

    r_perc = np.zeros(rtest.shape)
    r_perc[np.where(rtest >= 0)] = rtrans_gain(rtest[np.where(rtest >= 0)], gamma)
    r_perc[np.where(rtest < 0)] = rtrans_loss(rtest[np.where(rtest < 0)], lam,gamma)
    p_perc = prelec(ptest,alpha,delta)

    r_transform = [rtest, r_perc]
    p_transform = [ptest, p_perc]
    return r_transform,p_transform

def CPT_model(params,R):
    gamma, lam, alpha,delta,theta = params

    icert, igamble = 0, 1
    iloss, igain = 2, 3
    pcert = 1 # probability of outcome given certain action
    ploss = 0.5  # base prob of getting relative loss
    pgain = 1 - ploss # base gain prob

    # apply transformation on probability
    prelec = lambda p,a,d: np.exp(-a*np.power(-np.log(p),d))
    pcert = pcert # CPT not applied to probability here
    ploss = prelec(ploss,alpha,delta)
    pgain = prelec(pgain,alpha,delta)

    # apply transformations on reward
    rtrans_gain = lambda r,gamma1: np.power(r,gamma1)
    rtrans_loss = lambda r, lam1,gamma1: -lam1 * np.power(-r, gamma1)
    rcert = rtrans_gain(R[icert],gamma)     #np.power(R[icert],gamma)
    rgain = rtrans_gain(R[igain], gamma)
    rloss = rtrans_loss(R[iloss],lam,gamma) #-lam * np.power(-R[iloss], gamma)

    # transform utility
    u_reject = pcert * rcert
    u_accept = pgain*rgain + ploss*rloss
    Eai = np.zeros(2)
    Eai[icert] = u_reject
    Eai[igamble] = u_accept

    # create preference distribution (sensitivity to diff of Eu) returning p_accept
    Boltzmann_policy = lambda Eai1,theta1: (np.exp(theta1*Eai1)/np.sum(np.exp(theta1*Eai1)))[igamble]   # equivalant in binomial choice
    pi_accept = Boltzmann_policy(Eai,theta)

    # logit_policy = lambda Eai1, theta1: 1 / (1 + np.exp(-theta1 * (Eai1[igamble] - Eai1[icert])))  # equivalant in binomial choice
    # pi_accept = logit_policy(Eai, theta)

    return pi_accept

def objective(params,ai_obs,Rn):
    pi_accept = np.array([CPT_model(params, R) for R in Rn])
    was_accepted = ai_obs
    negLL = - np.sum(stats.norm.logpdf(was_accepted, pi_accept, params[-1]))
    return negLL

def generate_uniform_test_cases(r_cert=None,r_diff=None,p_penalty=None,seed=None):
    # Define indices
    nS,nA,nO = 4,2,100
    icert,igamble = 0,1
    # Define default parameters
    if seed is not None: np.random.seed(seed)
    p_penalty = 0.5 if p_penalty is None else p_penalty
    r_diff = np.random.choice(np.arange(10)) if r_diff is None else r_diff
    r_cert = 0 if r_cert is None else r_cert
    # define reward and transition
    R = np.array([r_cert,0,r_cert-r_diff,r_cert+r_diff])
    T = np.zeros([1,nA,nS])
    T[0, icert, :] = [1, 0, 0, 0]
    T[0, igamble, :] = [0,0,1-p_penalty,p_penalty]
    # Repeat for uniform observations
    Rn = np.zeros([nO, nS],dtype='float64')
    for obs in range(nO): Rn[obs] = R
    return Rn,T

if __name__ == "__main__":
    main()