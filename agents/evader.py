import numpy as np
from agents.policies import noisy_rational
from learning.generate_MDP import MDP_settings
from enviorment.utils import *
class EvaderDM(object):
    def __init__(self,Sk):
        self.rationality = 1
        self.rCatch = -MDP_settings['evader']['rCatch']
        self.scale_rDist = 2 # scaling factor of cumulative distance
        self.Sk = Sk
        self.idx = 2

        si_dmax = np.array([[1,1,0],[1,1,0],[5,5,0]])
        self.max_rDist = 1 # initialize no max dist scaling
        self.max_rDist = self.get_rDist(si_dmax)

    def get_unscaled_rDist(self,statei):
        order = 2
        scale_closer = 2
        d0 = math.dist(statei[0, 0:2], statei[self.idx, 0:2])
        d1 = math.dist(statei[1, 0:2], statei[self.idx, 0:2])
        r = pow(scale_closer * min(d0, d1), order) + pow(max(d0, d1), order)
        return r/self.max_rDist

    def get_reward(self,si):
        statei = self.Sk[si]
        if is_caught(statei,self.Sk): ri = self.rCatch
        else: ri = self.scale_rDist*self.get_unscaled_rDist(statei)
        return ri
    def get_pd(self,R):
        return noisy_rational(R,self.rationality)
    def choose(self,R):
        Ai = np.arange(np.size(R))
        pdA = self.get_pd(R)
        return np.random.choice(Ai,p=pdA)


if __name__ == "__main__":
    main()
