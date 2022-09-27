import numpy as np
# import matplotlib.pyplot as plt
from scipy.special import softmax

def noisy_rational(R,rationality):
    """softmax = np.exp(x) / sum(np.exp(x)"""
    R = np.array(R)
    epsilon = 1e-5 # arbitrary small number
    R[R==0] = epsilon
    pd = softmax(rationality*R)
    return pd



class CPT_Handler(object):
    def __init__(self,b,gamma,lam,alpha,delta,theta,**kwargs):
        """
        :param b: reference point
        :param gamma: view pos/neg reward as more pos/neg than actually are (symmetrical)
        :param lam: percieve losses with more weight than gains (lam>1) or vis-versa (lam<1)
        :param alpha: prelec parameter (how s-shaped prob weight function is)
        :param delta:
        :param theta: rationality (noisy rational decision)
        """
        self.b = b
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.delta = delta
        self.theta = theta
        self.epsilon = 1e-3 # small non-zero value
    def utility_weight(self,R):
        R= np.array(R).reshape([np.size(R),])
        b = np.mean(R) if self.b is None else self.b
        ipos = np.where(R-b >= 0)[0]
        ineg = np.where(R-b < 0)[0]
        R_perc = np.empty(np.shape(R))
        R_perc[ipos] = np.power(R[ipos]-b,self.gamma)
        R_perc[ineg] = -self.lam*np.power(np.abs(R[ineg]-b),self.gamma)
        return np.nan_to_num(R_perc)
    def prob_weight(self,p):
        """ Prelec probability weighting function """
        if np.any(p<0):
            raise Exception('Negative probabilities found in CPT Probability transform')
        p = np.array(p).reshape([np.size(p),])
        # p[np.where(p==0)] = self.epsilon
        p_perc = np.zeros(p.shape)
        for ai in range(len(p)):
            if p[ai] == 0: p_perc[ai]=0
            else: p_perc[ai] = np.exp(-self.alpha*np.power(-np.log(p[ai]),self.delta))
        # p_perc = np.exp(-self.alpha*np.power(-np.log(p),self.delta)) if p>0 else 0
        # p_perc[np.where(p == 0)] = 0
        return p_perc/np.sum(p_perc)#np.nan_to_num(p_perc)
    def expected_value_perc(self,R,T,si):
        """
        :param R: reward |nS|
        :param T: transition probability |nS x nA x nS|
        :return:
        """
        nS = np.shape(T)[0]
        nA = np.shape(T)[1]
        R_perc =  self.utility_weight(np.copy(R))
        T_perc = self.prob_weight(np.copy(T))
        Eai_perc = [np.sum(R_perc[si,ai] * T_perc[si, ai, :]) for ai in range(nA)]
        # Eai_perc = [np.sum(R_perc[si] * T_perc[si, ai, :]) for ai in range(nA)]
        # Eai = [np.sum(R[si] * T[si, ai, :]) for ai in range(nA)] # true expected value
        return Eai_perc
    # def noisy_rational(self,Eai):
    def boltzmann(self, Eai):
        return noisy_rational(Eai,rationality=self.theta)
        # V = np.copy(Eai)
        # V[np.where(V==0)] = self.epsilon
        # pd = np.exp(self.theta*Eai)/np.sum(self.theta*Eai)
        # return pd/np.sum(pd)
    def softmax(self,Eai): return self.boltzmann(Eai)
    def update_params(self,b,gamma,lam,alpha,delta,theta):
        self.b = b
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.delta = delta
        self.theta = theta
    def __call__(self, Q,T,si):
        """
        Assumes that dimensions for not the current state are removed
        :param Vsi: value of action ai
        :param Tsi: prob of sj given ai
        :return ichoice,pd: chosen action and prob dist over actions
        """
        nA = np.shape(Q)[1]
        V_perc = self.utility_weight(np.copy(Q[si]))
        ExpV = [np.sum(V_perc[ai]*T[si,ai,:]) for ai in range(nA)]
        pd = self.noisy_rational(ExpV)
        ichoice = np.random.choice(np.arange(nA),p=pd)
        # T_perc = self.prob_weight(np.copy(T))
        # Eai_perc = [np.sum(R_perc[si, ai] * T_perc[si, ai, :]) for ai in range(nA)]
        # Eai_perc = self.expected_value_perc([Vsi],[Tsi],si=0)
        # pd = self.noisy_rational(Eai_perc)
        # ichoice = np.random.choice(np.arange(nA),p=pd)
        return ichoice,pd

if __name__ == "__main__":
    R = [0,1,1,1,1]
    pd = noisy_rational(R,1)
