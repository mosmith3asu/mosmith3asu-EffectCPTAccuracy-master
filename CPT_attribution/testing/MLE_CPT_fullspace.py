import numpy as np
import matplotlib.pyplot as plt
import itertools
# import tqdm
# from CPT_attribution.testing.MLE_CPT_population import generate_uniform_test_cases
import seaborn as sns
from scipy.optimize import minimize

param_idx = {'gamma': 0, 'lam': 1, 'alpha': 2, 'delta': 3, 'theta': 4}
# fixed_params = {'lam':1.0,'theta': 1.0}
fixed_params = {'theta': 1.0}
def main_convergance():
    nObservations = 30
    nRepititions = 10

    param_idx = {'gamma': 0, 'lam': 1, 'alpha': 2, 'delta': 3, 'theta': 4}
    #fixed_params = {'theta': 1.0}  # dict of param name and value to be fixed
    Rn, T = generate_test_cases(r_diff=10)  # Generate decisions rewards and probability
    H_CPT = CPT_Model()
    X0 = get_human_model(**fixed_params)
    H_CPT.set_params(X0) # Define ground truth human model

    # Simulate human observations
    p = np.array([1, 0, 0.5, 0.5])  # prob of states
    ai_obs = np.zeros([len(Rn), 1])
    p0_accept = np.zeros([len(Rn), 1])
    for iobs, R in enumerate(Rn):
        p_gamble = H_CPT.get_pAccept(R, p)  # p of taking gable action
        p0_accept[iobs] = p_gamble
        ai_obs[iobs] = np.random.choice([0, 1], p=[1 - p_gamble, p_gamble])

    # recover parameters
    pdiff_accept = np.zeros([nObservations,1])
    CPT_tmp = CPT_Model()

    imu,isig,iRMSE = 0,1,2
    Hhat_CPT = CPT_Model()
    xstar_stats  = np.zeros([nObservations,5,3])
    for iobs in range(2,nObservations):
        xstar_rep = np.zeros([nRepititions,5])
        pdiff_rep = np.zeros([nRepititions,1])
        for irep in range(nRepititions):
            print(f'\robs={iobs}/{nObservations} \t rep = {irep}/{nRepititions}',end='')
            xsol = Hhat_CPT.MLE(ai_obs[:iobs],Rn[:iobs],p)
            xstar_rep[irep] = xsol

            CPT_tmp.set_params(xsol)
            phat_accept = ([CPT_tmp.get_pAccept(R,p) for R in Rn[:iobs]])
            pdiff_rep[irep] =np.mean(np.abs(p0_accept[:iobs] - phat_accept))
        pdiff_accept[iobs] = np.mean(pdiff_rep)

        xstar_stats[iobs,:,imu] = [np.mean(xstar_rep[:, iparam]) for iparam in range(5)]
        xstar_stats[iobs,:,isig] = [np.std(xstar_rep[:, iparam]) for iparam in range(5)]
        xstar_stats[iobs, :, iRMSE] = [
            np.sqrt(np.square(np.subtract(
            X0[iparam], xstar_rep[:, iparam]
            )).mean())  for iparam in range(5)
        ]

    fig,ax = plt.subplots(4,3)
    ax[0, 0].set_title(f'Recovered Mean and Std (rep={nRepititions})')
    ax[0, 1].set_title(f'Recovered RMSE (rep={nRepititions})')
    ax[0, 2].set_title(f'Mean Abs Difference in P(A) (rep={nRepititions})')
    ax[-1, 0].set_xlabel('N Observations')
    ax[-1, 1].set_xlabel('N Observations')
    for iax,key in enumerate(param_idx.keys()):
        if key == 'theta': break
        x_mu = xstar_stats[:, param_idx[key], imu]
        x_sig = xstar_stats[:, param_idx[key], isig]
        ax[iax,0].plot(x_mu, label='$\mu$')
        ax[iax,0].fill_between(np.arange(len(x_mu)), x_mu - x_sig, x_mu + x_sig, color='gray', alpha=0.2,label='std')
        ax[iax,0].hlines(X0[iax],xmin=0,xmax=nObservations-1,label='Truth',color='k',ls=':')
        ax[iax,0].legend()
        ax[iax,0].set_ylabel(key)

        ax[iax, 1].plot(xstar_stats[:, param_idx[key], iRMSE],label='RMSE')
        # ax[iax, 1].hlines(0, xmin=0, xmax=nObservations - 1, label='Zero Err', color='k', ls=':')
        ax[iax, 1].set_ylabel(key)
        ax[iax, 1].set_ylim([0,4])
        ax[iax, 1].legend()

        ax[iax, 2].plot(pdiff_accept, label='pdiff')
        # ax[iax, 1].hlines(0, xmin=0, xmax=nObservations - 1, label='Zero Err', color='k', ls=':')
        ax[iax, 2].set_ylabel(key)
        ax[iax, 2].set_ylim([0, 1])
        ax[iax, 2].legend()

    # plt.tight_layout()
    plt.show()
def main_paramspace():
    param_idx = {'gamma': 0, 'lam': 1, 'alpha': 2, 'delta': 3, 'theta': 4}
    #fixed_params = {'theta': 1.0}  # dict of param name and value to be fixed
    Rn, T = generate_test_cases(r_diff=10)  # Generate decisions rewards and probability

    H_CPT = CPT_Model()
    X0 = get_human_model(**fixed_params)
    H_CPT.set_params(X0) # Define ground truth human model

    p = np.array([1,0,0.5,0.5]) # prob of states
    # Simulate human observations
    ai_obs = np.zeros([len(Rn), 1])
    for iobs, R in enumerate(Rn):
        p_gamble = H_CPT.get_pAccept(R,p) # p of taking gable action
        ai_obs[iobs] = np.random.choice([0, 1], p=[1 - p_gamble, p_gamble])
    Hhat_CPT = CPT_Model()
    X_nLL = Hhat_CPT.nLL_paramspace(R,p,ai_obs,X0 = X0)
def main1():
    fig, ax = plt.subplots(2, 2)
    ax.reshape([2,2])
    # REWARD TRANSFORMATION
    CPT = CPT_Model()


    gamma0 = 0.9
    lam0 = 2
    alpha0 = .4
    delta0 = .7
    theta0 = 1
    params0 = [gamma0, lam0, alpha0, delta0, theta0]
    CPT.set_params(params0)
    #CPT.set_params(get_human_model())

    CPT.prelec = True
    rT0, pT0 = CPT.get_transform()  # prelec

    CPT.prelec = False
    rTstar, pTstar = CPT.get_transform() # prospect


    ax[0, 0].vlines(0, ymin=min(rT0[0]), ymax=max(rT0[0]), color='lightgrey', ls=':')
    ax[0, 0].hlines(0, xmin=min(rTstar[0]), xmax=max(rTstar[0]), color='lightgrey', ls=':')
    ax[0, 0].plot(rT0[0], rT0[1], label='$\mathcal{X}^{0}_{CPT}$')
    ax[0, 0].plot(rTstar[0], rTstar[1], label='$\mathcal{X}^{*}_{CPT}$')
    ax[0, 0].set_title('Reward Transformation')
    ax[0, 0].set_ylabel('Perceived Reward')
    ax[0, 0].set_xlabel('Relative Reward $(r-b)$')
    ax[0, 0].vlines(0, ymin=min(rT0[0]), ymax=max(rT0[0]), color='lightgrey', ls=':')
    ax[0, 0].hlines(0, xmin=min(rTstar[0]), xmax=max(rTstar[0]), color='lightgrey', ls=':')
    ax[0, 0].legend()

    ax[0, 1].plot(pT0[0], pT0[0], label='Optimal', color='lightgrey', ls=':')
    ax[0, 1].plot(pT0[0], pT0[1], label='Prelec',color = 'b')
    ax[0, 1].plot(pTstar[0], pTstar[1], label='Prospect$_{loss}$',color = 'r')
    ax[0, 1].plot(pTstar[0], pTstar[2], label='Prospect$_{gain}$',color = 'r',ls=':')
    ax[0, 1].set_title('Probability Transformation')
    ax[0, 1].set_ylabel('Perceived Probability')
    ax[0, 1].set_xlabel('Probability $(p)$')
    ax[0, 1].legend()
    plt.show()


class CPT_Model():
    def __init__(self):
        self.prelec = False
        self.rsym = True
        self.fr_prospect = lambda r,g,l: - l * np.power(np.abs(r), g)
        self.fp_prelec = lambda p, a, d: np.exp(-a * np.power(-np.log(p), d))
        self.fp_prospect = lambda p, w: np.power(p,w)/np.power(np.power(p,w)+ np.power(1-p,w),1/w)
        self.gamma, self.lam, self.alpha, self.delta, self.theta = 1,1,1,1,1 # defaults
        self.Xidx = {'gamma': 0, 'lam': 1, 'alpha': 2, 'delta': 3, 'theta': 4}
    def set_params(self,params):
        self.gamma, self.lam, self.alpha, self.delta, self.theta = params

    def _default_params(self,params):
        if params is None:
            gamma, lam, alpha, delta, theta = self.gamma, self.lam, self.alpha, self.delta, self.theta
        else: gamma, lam, alpha, delta, theta = params
        return  gamma, lam, alpha, delta, theta

    def get_rand_model(self,N=1,res=200):
        X = {}
        X['gamma']  = np.linspace(0.2, 0.8, res)
        X['lam']    = np.linspace(0.20, 5.00, res)# need to make |a>1|=|a<1
        if self.prelec:
            alpha_tmp   = np.linspace(1.01, 5.00, int(res / 2))  # need to make |a>1|=|a<1ha_tmp)
            X['alpha']  = np.append(np.power(alpha_tmp, -1), alpha_tmp)
            X['alpha']  = np.sort(X['alpha'])
        else:
            X['alpha'] = np.linspace(0.2, 0.8, res)
        X['delta']  = np.linspace(0.2, 0.8, res)
        X['theta'] = [1]
        samples = np.empty([N, len(X.keys())])
        for i in range(N):
            samples[i] = [np.random.choice(X[key]) for key in X.keys()]
        return samples,X

    def nLL_paramspace(self,R,p,ai_obs,nSamples = 10000,res=200,X0=None):
        ai_obs = np.array(ai_obs.flatten(),dtype='int8')
        param_names =['gamma','lam' ,'alpha','delta','theta']
        Xspace,X = self.get_rand_model(N=nSamples,res=res)
        negLL_M = np.empty([Xspace.shape[0],1])




        # Get negLL of all model samples
        for iM,params in enumerate(Xspace):
            for key in fixed_params.keys():
                params[param_idx[key]] = fixed_params[key]
            p_accept = self.get_pAccept(R,p,params)
            pM = np.zeros(ai_obs.shape)
            pM[ai_obs == 1] = p_accept
            pM[ai_obs == 0] = (1-p_accept)
            negLL_M[iM] = - np.sum(np.log(pM))
            # negLL_M[iM] = - np.sum(np.log(ai_obs*p_accept + (ai_obs)))

        data = {}
        for key in param_names: data[key] = []
        for iM in range(len(negLL_M)):
            gamma, lam, alpha, delta, theta = Xspace[iM,:]
            nLL = negLL_M[iM,0]
            data['gamma'].append([gamma,nLL])
            data['lam'].append([lam,nLL])
            data['alpha'].append([alpha,nLL])
            data['delta'].append([delta,nLL])
            data['theta'].append([theta,nLL])
            #tmp = np.array([[negLL_M[Xloc[key][ival]]] for ival, _ in enumerate(X[key])])
            # data[key] = tmp
        for key in data.keys():  data[key] = np.array(data[key])

        fig, ax = plt.subplots(5-1,2)
        ix = 0
        iy = 1
        labels = ['Value','$-LL$']
        ax[0, 0].set_title('Free Model Parameter Data')
        ax[0, 1].set_title('Free Model Paramater negLL Distribution ')
        ax[0, 0].set_xlabel(labels[ix])
        ax[0, 1].set_xlabel(labels[ix])
        for iax,key in enumerate(data.keys()):
            if key =='theta': break
            ax[iax, 0].scatter(data[key][:, ix], data[key][:, iy],s=1)
            ax[iax, 0].set_ylabel(f'({labels[iy]} | {key})')
            ax[iax, 1].set_ylabel(f'({labels[iy]} | {key})')

            x_param = data[key][:, ix]
            y_nLL = data[key][:, iy]
            nVals = len(X[key])
            win = 10
            x_mu = np.zeros(nVals)
            x_sig = np.zeros(nVals)
            # for vals in X[key][win:nVals - win]:
            for ival, val_cent in enumerate(X[key]):
                vals = X[key][max(0,ival-win):min(nVals,ival+win)]
                irange =  np.hstack([np.array(np.where(x_param == val)).flatten() for val in vals])
                x_mu[ival] = np.mean(y_nLL[irange])
                x_sig[ival] = np.std(y_nLL[irange])

            ax[iax, 1].plot(X[key],x_mu,label='$\mu$')
            ax[iax, 1].fill_between(X[key], x_mu - x_sig, x_mu + x_sig,  color='gray', alpha=0.2,label='$\pm \sigma$')
            # sns.kdeplot(ax=ax[iax,1], data={'val': data[key][:, 0], '-LL': data[key][:, 1]},  x='val', weights='-LL', cut=0)


            if X0 is not None:
                sz = 3
                ax[iax, 0].vlines(X0[iax], ymin=ax[iax, 0].get_ylim()[0],
                                  ymax=ax[iax, 0].get_ylim()[1],
                                  label='Truth', color='k', ls=':',lw=sz)
                ax[iax, 1].vlines(X0[iax],ymin = ax[iax,1].get_ylim()[0] ,
                                  ymax = ax[iax,1].get_ylim()[1],
                                  label='Truth',color='k',ls=':',lw=sz)
            ax[iax, 1].legend()
        #plt.tight_layout()
        plt.show()

    def get_transform(self,params=None):
        gamma, lam, alpha, delta, theta = self._default_params(params)

        res = 100
        p = np.linspace(0.01, 0.99, res)
        R = np.linspace(-10, 10, res)

        igain = np.where(R >= 0)
        iloss = np.where(R < 0)

        # reward transform
        r_perc = np.zeros(R.shape)
        r_perc[igain] = self.fr_prospect(R[igain],gamma, -1)
        r_perc[iloss] = self.fr_prospect(R[iloss], gamma,lam)
        r_transform = [R, r_perc]

        # probability transform
        if self.prelec:
            p_perc  = self.fp_prelec(p,alpha,delta)
            p_transform = [p, p_perc]
        else:
            # p_perc = np.zeros(p.shape)
            # p_perc[igain] = self.fp_prospect(p[igain], alpha)
            # p_perc[iloss] = self.fp_prospect(p[iloss], delta)
            p_gain = self.fp_prospect(p, alpha)
            p_loss = self.fp_prospect(p, delta)
            p_transform = [p, p_loss,p_gain]
        return r_transform, p_transform
    def get_pAccept(self,R,p,params=None):
        gamma, lam, alpha, delta, theta = self._default_params(params)
        ireject, igamble = 0, 1
        icert,iloss,igain = 0,2,3
        ri_gain = np.where(R >= 0)
        ri_loss = np.where(R < 0)

        # reward transform
        r_perc = np.zeros(R.shape)
        r_perc[ri_gain] = self.fr_prospect(R[ri_gain], gamma, -1)
        r_perc[ri_loss] = self.fr_prospect(R[ri_loss], gamma, lam)

        # probability transform
        if self.prelec:
            p_perc  = self.fp_prelec(p,alpha,delta)
        else:
            p_perc = np.zeros(p.shape)
            p_perc[ri_gain] = self.fp_prospect(p[ri_gain], alpha)
            p_perc[ri_loss] = self.fp_prospect(p[ri_loss], delta)


        u_reject = p_perc[icert] * r_perc[icert]
        u_accept = p_perc[igain] * r_perc[igain] + p_perc[iloss] * r_perc[iloss]
        Eai = np.zeros(2)
        Eai[ireject] = u_reject
        Eai[igamble] = u_accept

        Boltzmann_policy = lambda Eai1, theta1: (np.exp(theta1 * Eai1) / np.sum(np.exp(theta1 * Eai1)))[igamble]  # equivalant in binomial choice
        pi_accept = Boltzmann_policy(Eai, theta)
        return pi_accept

    def objective_nLL(self,params,ai_obs,Rn,p):
        p_accept = np.array([self.get_pAccept(R, p, params) for R in Rn])
        pM = np.zeros(ai_obs.shape)
        p_accept[ai_obs.flatten()==0] = (1 - p_accept[ai_obs.flatten()==0])
        negLL = - np.sum(np.log(pM))
        return negLL

    def MLE(self,ai_obs,Rn,p):
        def get_opt_params(verbose=False, **kwargs):

            gamma1, gamma_bnds = 0.87, (0.01, 0.99)  # gamma1, gamma_bnds = 0.87, (0.01,0.99)
            lam1, lam_bnds = 1.27, (0.20, 5.00)  # lam1,   lam_bnds   = 1.27, (1.00,5.00)
            alpha1, alpha_bnds = 2.10, (0.20, 5.00)  # alpha1, alpha_bnds = 2.10, (1.00,5.00)
            delta1, delta_bnds = 0.80, (0.01, 0.99)  # delta1, delta_bnds = 0.80, (0.01,0.99)
            theta1, theta_bnds = 1.00, (0.50, 5.00)  # theta1, theta_bnds = 1.00, (1.00,5.00)

            # guess = [gamma1, lam1, alpha1, delta1, theta1]
            guess,_ = self.get_rand_model()
            guess = guess.flatten()

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

            return guess, bnds, cons
        # fixed_params = {'alpha':1.0,'theta': 1.0}
        guess, bnds, cons = get_opt_params(**fixed_params)  # inputs fix params as constant
        result = minimize(self.objective_nLL, np.array(guess),
                          args=(np.array(ai_obs), Rn,p),
                          constraints=cons, bounds=bnds,
                          #method='Nelder-Mead'#, options={'disp':True}
                          )
        xstar = result['x']
        return xstar

def get_human_model(imodel=0,verbose=False,**kwargs):
    param_idx = {'gamma': 0, 'lam': 1, 'alpha': 2, 'delta': 3, 'theta': 4}
    if imodel==0: # get random human model within reasonable bounds
        res = 100
        gamma0 = np.random.choice(np.linspace(0.2,0.8,res))
        lam0   = np.random.choice(np.linspace(0.20,5.00,res))
        # alpha_tmp = np.linspace(1.01,5.00,int(res/2)) # need to make |a>1|=|a<1| for fair sampling of inverse behaviors
        # alpha0 = np.random.choice(np.append(np.power(alpha_tmp,-1),alpha_tmp))
        alpha0 = np.random.choice(np.linspace(0.2,0.8,res))
        delta0 = np.random.choice(np.linspace(0.2,0.8,res))
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

def generate_test_cases(r_cert=None,r_diff=None,p_penalty=None,seed=None):
    # Define indices
    nS,nA,nO = 4,2,100
    icert,igamble = 0,1
    # Define default parameters
    if seed is not None: np.random.seed(seed)
    p_penalty = 0.5 if p_penalty is None else p_penalty
    r_possible = np.linspace(0,r_diff,25)
    r_cert = 0

    T = np.zeros([1,nA,nS])
    T[0, icert, :] = [1, 0, 0, 0]
    T[0, igamble, :] = [0,0,1-p_penalty,p_penalty]
    r_gain = np.random.choice(r_possible)
    r_loss = np.random.choice(-r_possible)
    # Repeat for uniform observations
    Rn = np.zeros([nO, nS],dtype='float64')
    for obs in range(nO):
        # r_gain = np.random.choice(r_possible)
        # r_loss = np.random.choice(-r_possible)
        R = np.array([r_cert, 0, r_loss, r_gain])
        Rn[obs] = R
    return Rn,T
if __name__ == "__main__":
    main_convergance()
    main_paramspace()