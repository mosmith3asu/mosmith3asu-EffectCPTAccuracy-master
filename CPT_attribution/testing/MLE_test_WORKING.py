import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from math import pi
# np.random.seed(0)
def main():

    N = 50
    t = np.linspace(0,5,N)

    # params = np.array([2, 18, 2, 4])
    bnds = [[0, 100], [0, 100],[0, 100], [0, 20]] # [a,b,c=0,sig]
    params = np.array([np.random.choice(np.linspace(bnds[0][0]+1,bnds[0][1],20)),
                       np.random.choice(np.linspace(bnds[1][0]+1,bnds[1][1],20)),
                       np.random.choice(np.linspace(bnds[2][0]+1,bnds[2][1],20)),
                       np.random.choice(np.linspace(bnds[3][0]+1,bnds[3][1],20))])


    # Sine function ===========
    # params = np.array([ pi/8,  pi/4, 1.0, 0.5]) #[freq,phase,amp,noise]
    # params = np.array([ pi,  pi/4, 1.22, 0.1]) #[freq,phase,amp,noise]
    # bnds = [[0, pi], [0, 2 * pi], [0, 5], [0, 2]]
    yGT  = sample_model(params,t,noise=False)
    yobs = sample_model(params,t)
    plt.plot(t,yGT,label='Ground Truth')
    plt.scatter(t,yobs,label='Samples')
    # plt.show()

    MLE = MLE_example()
    MLE.bnds = bnds
    params_star = MLE.run(t,yobs,
                          guess=None,#params,
                          ground_truth = params)
    ystar = sample_model(params_star, t,noise=False)

    plt.plot(t,ystar, label='MLE Model')
    # plt.scatter(np.arange(N), ystar_samples, label='MLE Samples')

    # print(f'x0={params}')
    # print(f'x*={params_star}')
    plt.legend()
    plt.show()
def sample_model(params,t,noise=True):
    # Model
    y = params[0]*np.power(t,2) + params[1]*t + params[2] # y = amp*np.sin(freq*t+phase)
    e = np.random.normal(0, params[-1], len(t)) if noise else 0  # Noise
    return y + e

class MLE_example():
    def __init__(self):
        self.nparams = 4
        self.cons = [{ 'type': 'ineq', 'fun': lambda params: params[0]-0.01},
                     { 'type': 'ineq', 'fun': lambda params: params[1]-0.01},
                     {'type': 'ineq', 'fun': lambda params: params[2]-0.01},
                     {'type': 'ineq', 'fun': lambda params: params[3]-0.01},
                     ]

        # self.bnds = [[0,pi],[0,2*pi],[0,5],[0,2]]
        self.bnds = []
        # self._def_guess = [np.random.choice(
        #     np.linspace(self.bnds[ip][0], self.bnds[ip][1], 20))
        #     for ip in  range(self.nparams)]


        # self.solver = 'L-BFGS-B'
        # self.solver = 'BFGS'
        self.solver = 'SLSQP'

        # self.solver = 'Nelder-Mead'
        self.samples = []
        self.x_star = []

    def c_sig(self,parameters):
        sigma = parameters[3]
        return sigma
    def objective(self,parameters,t,y_obs):
        y_exp = sample_model(parameters,t,noise=False)
        #std_dev = np.std(y_obs-y_exp)
        std_dev = parameters[-1]
        neg_LL = - np.sum(stats.norm.logpdf(y_obs,y_exp,std_dev))


        # negLL = - np.sum(np.log(stats.norm.pdf(y_obs-y_exp, loc=0, scale=sigma)))
        return neg_LL


    def run(self,t,yobs,guess=None,ground_truth=None):
        # self._def_guess = [np.random.choice(
        #     np.linspace(self.bnds[ip][0], self.bnds[ip][1], 20))
        #     for ip in range(self.nparams)]
        self._def_guess = [np.mean(self.bnds[ip]) for ip in range(self.nparams)]
        guess = self._def_guess if guess is None else guess

        N = yobs.size

        # self.guess = np.array([pi/4,pi/4,1,1])
        print(f'Running MLE:')
        print(f'\t| N samples = {N}')
        print(f'\t| Guess     = {np.round(guess, 2)}')
        if ground_truth is not None: print(f'\t| Truth     = {np.round(ground_truth, 2)}')



        self.samples = np.array(yobs)
        result = minimize(self.objective, guess,
                          constraints=self.cons,
                          args=(t,self.samples,),
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
