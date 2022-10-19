# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy.optimize import minimize
from scipy import stats
import pandas
import seaborn as sns
def simulate_MLE_recovery(fixed_params=None):
    fixed_params = {} if fixed_params is None else fixed_params
    #######################################################
    # Human and Environment ###############################
    Rn, T = generate_uniform_test_cases(r_diff=10)  # Generate decisions rewards and probability
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
    # Report Simulation ###################################
    col_labels = ['gamma','lam','alpha','delta','theta']
    rowdata = {'Bounds':[(round(bnd[0],2),round(bnd[1],2)) for bnd in bnds],
               'Guess': guess,
               'True': params0,
               'MLE': np.round(xstar,2),
               'err': list((np.array(params0)-np.array(xstar).round(2))) }
    df = pandas.DataFrame.from_dict(rowdata,orient='index',columns=col_labels)
    df = df.round(2)
    print(df)

    # Package simulation info
    info = {}
    info['Rn'] = Rn
    info['guess'] = guess
    info['bnds'] = bnds
    info['cons'] = cons
    info['xtrue'] = params0
    info['xstar'] = xstar
    info['xerror'] = np.array(guess)-np.array(xstar)
    return info
def report_simulation_stats(data):
    # Display Settings
    width = 0.35  # the width of the bars
    nrows,ncols = 2,1
    labels = ['$\\gamma$\n(diminishing rewards)', '$\\lambda$\n(loss aversion)',
              '$\\alpha$\n(over/underest. prob.)', '$\\delta$\n(diminishing prob.)',
              '$\\theta$\n(rationality)']
    xlabel = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots(nrows, ncols)
    ax = ax.reshape([nrows,ncols])

    # Calculate reported population stats
    settings = {}
    var = 'xtrue'
    loc = [0, 0]
    title = 'Ground Truth Params'
    color = 'grey'
    mu = np.mean(data[var],axis=0)
    std= np.std(data[var],axis=0)
    settings[var] = [mu, std, loc, title, color]

    var = 'xstar'
    loc = [0, 1]
    title = 'MLE Recovered Params'
    color = 'g'
    mu = np.mean(data[var], axis=0)
    std = np.std(data[var], axis=0)
    settings[var] = [mu, std, loc, title, color]

    var = 'xerr'
    loc = [1, 0]
    title = 'Parameter Estimation Error'
    color = 'r'
    mu = np.mean(data[var], axis=0)
    std = np.std(data[var], axis=0)
    settings[var] = [mu, std, loc, title, color]

    r,c=0,0
    rects1 = ax[r,c].bar(xlabel - width / 2, settings['xtrue'][0], width, label='$\mathcal{X}^{0}_{CPT}$',color='grey',
                         yerr= settings['xtrue'][1],align='center', alpha=0.5, ecolor='black', capsize=10)
    rects2 = ax[r,c].bar(xlabel + width / 2, settings['xstar'][0], width, label='$\mathcal{X}^{*}_{CPT}$',color='g',
                         yerr= settings['xstar'][1],align='center', alpha=0.5, ecolor='black', capsize=10)
    ax[r, c].set_ylabel('Parameter Value')
    ax[r, c].set_title('Ground Truth and MLE Recovered CPT Parameters')
    ax[r, c].set_xticks(xlabel, labels)
    ax[r, c].legend()

    r, c = 1, 0
    rects = ax[r, c].bar(xlabel, settings['xerr'][0], width, color='r', label='$\epsilon(\mathcal{X}_{CPT})$',
                         yerr=settings['xerr'][1],align='center', alpha=0.5, ecolor='black', capsize=10)
    ax[r, c].set_ylabel('Error')
    ax[r, c].set_title('CPT Parameter Recovery Error')
    ax[r, c].set_xticks(xlabel, labels)
    ax[r, c].hlines(0, xmin=np.min(xlabel)-width,xmax=np.max(xlabel)+width,color = 'k',ls=':')
    ax[r, c].legend()
    # ax[r,c].bar_label(rects1, padding=3)
    # ax[r,c].bar_label(rects2, padding=3)


    ############################################
    # DENSITY PLOTS ###########################
    binwidth = 3
    enbl_hist = True
    enbl_kde = False
    lw = 2
    nParams =  data['xtrue'].shape[1]
    nrows, ncols =3,2
    fig2, ax2 = plt.subplots(nrows, ncols)
    ax2 = ax2.reshape([nrows, ncols])
    names = ['\gamma', '\lambda', '\\alpha', '\delta', '\\theta']

    r,c = 0,0
    for ivar in range(nParams):
        if c == ncols: r,c = r+1,0
        sns.distplot(data['xtrue'][:, ivar], ax=ax2[r, c], label=f'${names[ivar]}^0$', color='grey', hist=enbl_hist,
                     hist_kws={'edgecolor': 'grey'}, kde=enbl_kde, kde_kws={'linewidth': lw}, bins=int(100 / binwidth), )
        sns.distplot(data['xstar'][:, ivar], ax=ax2[r, c], label=f'${names[ivar]}^*$', color='g', hist=enbl_hist,
                     hist_kws={'edgecolor': 'g'}, kde=enbl_kde, kde_kws={'linewidth': lw}, bins=int(100 / binwidth), )
        ax2[r, c].set_xlim([0,np.max(data['xtrue'][:,ivar])])
        ax2[r,c].set_title(f'${names[ivar]}$ Density Functions')
        ax2[r, c].legend()
        c += 1
    fig2.tight_layout()

def main(nSimulations = 10000):
    global param_idx
    #######################################################
    # Define Global settings ##############################
    param_idx = {'gamma': 0, 'lam': 1, 'alpha': 2, 'delta': 3, 'theta': 4}
    fixed_params = {'theta': 1.0} # dict of param name and value to be fixed
    nParams = len(param_idx.keys())

    #######################################################
    # Simulate Population MLE recovery ####################
    data = {}
    data['xtrue'] = np.empty([nSimulations,nParams])
    data['xstar'] = np.empty([nSimulations, nParams])
    data['xerr']  = np.empty([nSimulations, nParams])

    for isim in range(nSimulations):
        print(f'\n###### Simulation {isim} ######')
        result = simulate_MLE_recovery(fixed_params)
        data['xtrue'][isim] = result['xtrue']
        data['xstar'][isim] = result['xstar']
        data['xerr'][isim]  = result['xerror']

    report_simulation_stats(data)
    plt.show()
def get_human_model(imodel=0,verbose=False,**kwargs):
    global param_idx
    if imodel==0: # get random human model within reasonable bounds
        res = 100
        gamma0 = np.random.choice(np.linspace(0.01,0.99,res))
        lam0   = np.random.choice(np.linspace(0.20,5.00,res))
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
    if len(kwargs.keys())>0 and verbose:
        print(f'\nFixing H Model Params:')
    for key in kwargs:
        if verbose: print(f'\t|{key} = {kwargs[key]}')
        params0[param_idx[key]] =  kwargs[key]
    return params0
def get_opt_params(verbose=False,**kwargs):
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
    if len(kwargs.keys()) > 0 and verbose:
        print(f'Fixing MLE Params:')
    for key in kwargs:
        if verbose: print(f'\t|{key} = {kwargs[key]}')
        bnds[param_idx[key]] = (kwargs[key] - eps, kwargs[key] + eps)
        guess[param_idx[key]] = kwargs[key]

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