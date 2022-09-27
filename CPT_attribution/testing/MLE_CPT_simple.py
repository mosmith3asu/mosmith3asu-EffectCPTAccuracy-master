# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy.optimize import minimize

def main():
    # Genearte envioremnt
    Rn,T = generate_test_cases()

    # Simulate human observaations
    gamma0 = 0.5
    lam0 = 1.5
    theta0 = 0.5
    params0 = [gamma0,lam0,theta0]
    # params0 = [gamma0, theta0]
    ai_obs = [CPT_model(params0,R,choose=True) for R in Rn]

    # optimize
    # guess = np.ones(len(params0))
    # guess[-1] = 1.1*theta0

    guess = [0.83,1.4,2.57]
    cons = {'type':'ineq','fun': lambda params: params[-1] }
    if len(params0)==3: bnds = [(0.2,2),(0.2,5),(0,1)]
    else:  bnds = [(0.2,2),(0.01,5)]
    result = minimize(objective, guess,
                      args=(np.array(ai_obs),Rn,),
                      tol=1e-10,
                      constraints=cons,
                      bounds=bnds,
                      # method=self.solver,
                      # options={'disp':True}
                      )
    xstar = result['x']
    print(f'True = {np.array(params0).round(2)}')
    print(f'MLE  = {np.array(xstar).round(2)}')

    fig, ax = plt.subplots(2, 1)
    GAMMAS = np.linspace(0,1,100)
    MUS = [0.5,1,3]
    for mu in MUS:
        P_ACCEPT0 = []
        U_DIFF0 = []
        P_ACCEPT = []
        U_DIFF = []
        if len(params0)==3: params_tmp = [gamma0, lam0, mu]
        else: params_tmp = [gamma0, mu]
        for R in Rn:
            p_accept0,u_diff0 = CPT_model(params_tmp,R,get_udiff=True)
            P_ACCEPT0.append(p_accept0)
            U_DIFF0.append(u_diff0)

            p_accept, u_diff = CPT_model(xstar, R, get_udiff=True)
            P_ACCEPT.append(p_accept)
            U_DIFF.append(u_diff)
        ax[0].scatter(U_DIFF0,P_ACCEPT0, label=f'$mu={mu}$')
    ax[1].scatter(U_DIFF, P_ACCEPT, label=f'$recovered$')

    ax[0].legend()
    ax[0].set_xlabel("Difference in Subjective Utility \n $E\{u(gamble)\}-E\{u(cert)\}}$")
    ax[0].set_ylabel("Prob of Accepting \nGamble p(gamble)$")
    plt.show()

def CPT_model(params,R,choose=False,get_udiff=False):
    if len(params)==2:
        gamma, rationality = params
        lam = 1
    else:
        gamma, lam, rationality = params

    icert, igamble = 0, 1
    iloss, igain = 2, 3
    ploss = 0.5  # [0,0,0.5,0.5]
    pgain = 1 - ploss

    # transform utility
    u_reject = np.power(R[icert],gamma)
    u_accept = np.sum([pgain * np.power(R[iloss], gamma),
                       ploss * (lam * np.power(R[iloss], gamma))
                       ])
    u_diff = u_accept - u_reject  # difference in subjective utility of the two choices
    pi_accept = 1 / (1 + np.exp(-rationality * u_diff))
    choice = np.random.choice([icert,igamble],p=[1-pi_accept,pi_accept])
    if np.all(choose and get_udiff): return choice,u_diff
    elif choose: return choice
    elif get_udiff: return pi_accept,u_diff
    else: return pi_accept

def objective(params,ai_obs,Rn):
    N = len(ai_obs)
    pi_accept =  np.array([CPT_model(params,R) for R in Rn])
    yi = ai_obs # y = {1 if choose gamble and 0 if choose cert}
    # trial_liklihood = np.dot(np.power(pi_accept,yi) * np.power(1-pi_accept,1-yi))
    negLL = - np.sum(yi * np.log(pi_accept) + (1 - yi)*(np.log(1 -pi_accept)))
    return negLL


    # Calc expected value
    # Eai = np.zeros([N, 2])
    # Eai[:,icert] = np.sum(T[0, icert, :]*R,axis=1)
    # Eai[:, igamble] = np.sum(T[0, igamble, :] * R,axis=1)




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
if __name__ == "__main__":
    main()
