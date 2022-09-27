import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from math import pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
# np.random.seed(0)
def main():




    # Define Model ========================
    N = 50
    t = np.linspace(0, 5, N)

    bnds = [[0, 100], [0, 100], [0, 20]] # [a,b,c=0,sig]
    # params = np.array([2, 18, 2, 4])
    params = np.array([np.random.choice(np.linspace(bnds[0][0]+1,bnds[0][1],20)),
                       np.random.choice(np.linspace(bnds[1][0]+1,bnds[1][1],20)),
                       np.random.choice(np.linspace(bnds[2][0]+1,bnds[2][1],20)),
                       # np.random.choice(np.linspace(bnds[3][0]+1,bnds[3][1],20))
                       ])

    # Display Ground Truth and Obs =========
    yGT  = sample_model(params,t,noise=False)
    yobs = sample_model(params,t)
    plt.plot(t,yGT,label='Ground Truth')
    plt.scatter(t,yobs,label='Samples')
    # plt.show()

    # Perform MLE ==============
    MLE = MLE_example(nx=3)
    MLE.bnds = bnds
    params_star = MLE.run(t,yobs, guess=None, ground_truth = params)

    # Plot Results ===========
    ystar = sample_model(params_star, t,noise=False)
    plt.plot(t,ystar, label='MLE Model')

    # Get parameter PDF ========
    res = 20
    x1 = np.linspace(bnds[0][0], bnds[0][1],res)
    x2 = np.linspace(bnds[1][0], bnds[1][1], res)
    x3 = 5
    xx1, xx2 = np.meshgrid(x1, x2)
    z = np.zeros([res,res])
    for i1,ix1 in enumerate(x1):
        for i2,ix2 in enumerate(x2):
            # z[i1, i2] = MLE.log_liklihood([ix1, ix2, x3], t, yobs)
            z[i1, i2] = MLE.liklihood([ix1, ix2, x3], t, yobs)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx1, xx2, z, cmap=cm.jet)
    ax.set_xlabel('$ax^2(t)$')
    ax.set_ylabel('$bx(t)$')
    ax.set_title('Liklihood')

    plt.legend()
    plt.show()
def sample_model(params,t,noise=True):
    y = params[0]*np.power(t,2) + params[1]*t # model
    e = np.random.normal(0, params[-1], len(t)) if noise else 0  # Noise
    return y + e

class MLE_example():
    def __init__(self,nx):
        self.nparams = nx
        self.cons = [{ 'type': 'ineq', 'fun': lambda params: params[0]-0.01},
                     { 'type': 'ineq', 'fun': lambda params: params[1]-0.01},
                     {'type': 'ineq', 'fun': lambda params: params[2]-0.01},
                     # {'type': 'ineq', 'fun': lambda params: params[3]-0.01},
                     ]

        # Define Aux Params ==============
        self.bnds = []
        self._def_guess = []
        self.samples = []
        self.x_star = []
        # Define Solver ===============
        self.solver = None
        # self.solver = 'L-BFGS-B'
        # self.solver = 'BFGS'
        # self.solver = 'SLSQP'
        # self.solver = 'Nelder-Mead'

    def log_liklihood(self,parameters,t,y_obs):
        y_exp = sample_model(parameters,t,noise=False)
        pdf_L =  np.sum(stats.norm.logpdf(y_obs, y_exp, parameters[-1]))
        return pdf_L
    def liklihood(self,parameters,t,y_obs):
        y_exp = sample_model(parameters,t,noise=False)
        pdf_L = np.sum(stats.norm.pdf(y_obs, y_exp, parameters[-1]))
        return pdf_L
    def objective(self,parameters,t,y_obs):
        y_exp = sample_model(parameters,t,noise=False)
        std_dev = parameters[-1]  #std_dev = np.std(y_obs-y_exp)
        neg_LL = - np.sum(stats.norm.logpdf(y_obs,y_exp,std_dev))
        # negLL = - np.sum(np.log(stats.norm.pdf(y_obs-y_exp, loc=0, scale=sigma)))
        return neg_LL


    def run(self,t,yobs,guess=None,ground_truth=None):
        self._def_guess = [np.mean(self.bnds[ip]) for ip in range(self.nparams)]
        guess = self._def_guess if guess is None else guess
        print(f'Running MLE:')
        print(f'\t| N samples = {yobs.size}')
        print(f'\t| Guess     = {np.round(guess, 2)}')
        if ground_truth is not None: print(f'\t| Truth     = {np.round(ground_truth, 2)}')

        result = minimize(self.objective, guess,
                          constraints=self.cons,
                          args=(t,np.array(yobs),),
                          tol = 1e-10,
                          bounds=self.bnds,
                          method=self.solver,
                          # options={'disp':True}
                          )
        self.x_star = result['x']

        print(f'\t| Recovered = {np.round(self.x_star,2)}')
        if ground_truth is not None:  print(f'\t| %Error    = {np.round(100*(ground_truth - self.x_star)/ground_truth, 2)}')
        return self.x_star

if __name__ == "__main__":
    main()
