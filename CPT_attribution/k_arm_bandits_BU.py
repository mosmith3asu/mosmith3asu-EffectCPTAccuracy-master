import numpy as np
import pandas
from agents.policies import noisy_rational,CPT_Handler
from functools import partial
from itertools import product as cartesian_product
import matplotlib.pyplot as plt


class uniform_kArmedBandits(object):
    def __init__(self):
        """
        :param nA: number of actions
        :param nO: number of outcomes
        """

        # Define indexes
        self.iP = 0     # prob index
        self.iR = 1     # reward index
        self.iOp = 0    # index for "positive" outcome fo ai: o^{+}
        self.iOn = 1    # index for "negative" outcome fo ai: o^{-}

        self.boltzmann_policy = noisy_rational # optimal agent boltzmann policy
        # self.boltzmann_policy = partial(noisy_rational, rationality=1)  # optimal agent boltzmann policy

        # Define static params
        self.E0 = 0                     # uniform expected reward across all actions
        self.p0 = 0.5                   # baseline probability to modify to enforce indifference
        self.r0 = 10                    # baseline reward (+/-) to enforce indifference
        self.O = [self.iOp, self.iOn]   # outcomes of each action (pos or neg)
        self.Info = [self.iP,self.iR]   # information given for each action

        # define game
        self.nA = 0                                 # number of initial actions
        self.nO = len(self.O)                       # number of outcomes
        self.nInfo = len(self.Info)                 # number of information vars for each action
        self.Eai = []                               # expected value for each action
        self.G = np.zeros([0,self.nO,self.nInfo])   # the game form

        # Populate game
        self._phats = {} # \in [-1,1]
        self._phats['a_0'] = 0
        self._phats['a_pc'] = 0.5
        self._phats['a_p+'] = 0.25
        self._phats['a_p-'] = -0.25
        for ai, key in enumerate(self._phats.keys()):  self.add_gamble(phat=self._phats[key])

        # Comparison Modes
        self.COMPARISON_MODES = ['population','binary']
        self._cmp_mode = self.COMPARISON_MODES[1]

        self._icmp0 = 0                                     # compare against action index
        self._icmp0_def = np.NAN                            # default value for compared action
        self._cmps = list(range(self.nA)).pop(self._icmp0)  # comparison indexes
        self._binary_comparisons = [[self._icmp0,ai] for ai in range(self.nA-1)] # index comparison handlers
        if self._cmp_mode == 'binary': self.uniform_preference = np.ones(self.nA) / 2  # indifference between actions
        elif self._cmp_mode == 'population': self.uniform_preference = np.ones(self.nA) / self.nA  # indifference between actions
        else: raise Exception('in uniform_kArmedBandits >> unknown comparison mode')

    #######################################################
    # GAME TOOLS ##########################################
    def add_gamble(self,phat):
        """ E(ai|p0) = [{p0+phat,r0(1-phat), {p0-phat,r0(1+phat)} """
        Gai = np.zeros([1, self.nO, len([self.iP, self.iR])]) # temp game line

        # "positive" outcome fo ai: o^{+}
        Gai[0,self.iOp,self.iP] = self.p0 + phat
        scale =((self.p0 - phat)/self.p0)
        Gai[0,self.iOp,self.iR] = self.r0*scale

        # "negative" outcome fo ai: o^{-}
        Gai[0, self.iOn, self.iP] = self.p0 - phat
        scale = ((self.p0 + phat) / self.p0)
        Gai[0, self.iOn, self.iR] = -self.r0 * scale

        # Append new action
        self.G = np.append(self.G,Gai,axis=0)
        self.nA = np.shape(self.G)[0]
    def preview_game(self):
        print(f'\n\n')
        print(f'###########################################################')
        print(f'################# UNIFORM k-Armed Bandits #################')
        print(f'###########################################################')
        data_dict = {}
        col_headers = ['Eai',' ','p+', 'r+','[V+]',' ','p-', 'r-','[V+]']
        sig_dig = 2
        for ai, key in enumerate(self._phats.keys()):
            pinfo = self.G[ai, self.iOp, :]             # positive info
            ninfo = self.G[ai, self.iOn, :]             # negative info
            Ep = np.product(self.G[ai, self.iOp, :])    # positive expected reward
            En = np.product(self.G[ai, self.iOn, :])    # negative expected reward
            Eai = Ep+En
            data_dict[key] = [Eai, ' ',pinfo[self.iP], pinfo[self.iR],Ep, ' ',ninfo[self.iP], ninfo[self.iR],En]
        df = pandas.DataFrame.from_dict(data_dict, orient='index', columns=col_headers)
        print(df.round(sig_dig))
    def evaluate(self,G=None):
        """ Evaluate expected value for each action ai"""
        if G is None: G=self.G
        Eai = np.sum(np.prod(G[:,:,:],axis = 2),axis=1)
        return Eai

    #######################################################
    # POLICY PREFERENCES ##################################
    def preference_anomaly(self,CPT_params):
        cpt_pref = self.CPT_preferences(CPT_params)
        anomaly = cpt_pref - self.uniform_preference
        return anomaly
    def optimal_preferences(self,rationality=1):
        rat = rationality
        Eai = self.evaluate(G=self.G)
        if self._cmp_mode == 'binary':
            prefs = [self._icmp0_def] + [self.boltzmann_policy(Eai[icmp],rationality=rat)[1] for icmp in self._binary_comparisons]
        elif self._cmp_mode == 'population':  prefs = self.boltzmann_policy(Eai,rationality=rat)
        else: raise Exception('in uniform_kArmedBandits >> unknown comparison mode')
        return np.array(prefs)
    def CPT_preferences(self,CPT_params):
        CPT_policy = CPT_Handler(**CPT_params)
        G_perc = np.empty(np.shape(self.G)) # perceived game
        cartesian_product(*[np.arange(self.nA),self.O])
        for ai,iO in cartesian_product(*[np.arange(self.nA),self.O]):
            G_perc[ai,iO,self.iR] = CPT_policy.utility_weight(self.G[ai,iO,self.iR])
            G_perc[ai,iO,self.iP] = CPT_policy.prob_weight(self.G[ai,iO,self.iP])
        Eai_perc = self.evaluate(G=G_perc)

        if self._cmp_mode == 'binary':
            prefs = [self._icmp0_def] + [CPT_policy.boltzmann(Eai_perc[icmp])[1] for icmp in self._binary_comparisons]
        elif self._cmp_mode == 'population':  prefs = CPT_policy.boltzmann(Eai_perc)
        else: raise Exception('in uniform_kArmedBandits >> unknown comparison mode')
        return np.array(prefs)


    def plot_preferences(self,ax,vals,is_anomaly=False):

        labels =list(self._phats.keys())
        unf_pref = self.uniform_preference
        width = 0.35  # the width of the bars
        pad = 3

        if self._cmp_mode == 'binary':
            labels = labels[1:]
            vals = vals[1:]
            unf_pref = unf_pref[1:]
        else: raise Exception('unknown comparison mode')
        x = np.arange(len(labels))  # the label locations


        if is_anomaly:
            # Only plot the anomaly
            rects_cpt = ax.bar(x + width/2, vals, width, label='CPT')
            ax.bar_label(rects_cpt, padding=pad)
            ax.set_title('Preference Anomaly')
        else:
            # plot against uniform preference
            rects_opt = ax.bar(x - width / 2, unf_pref, width, label='Opt')
            rects_cpt = ax.bar(x + width / 2, vals, width, label='CPT')
            ax.bar_label(rects_opt, padding=pad)
            ax.bar_label(rects_cpt, padding=pad)
            ax.set_title('Optimal vs CPT Preference')

        ax.set_ylabel('Actions')
        ax.set_xticks(x, labels)
        ax.legend()


class CPT_Models(object):
    def __init__(self,num0 = 3):
        self.bounds = {}
        self.bounds['b']     = 0  # reference point
        self.bounds['gamma'] = {'start':0.20, 'stop':0.80, 'num':num0}  # diminishing return gain
        self.bounds['lam']   = {'start':0.01, 'stop':10.0, 'num':num0}  # loss aversion
        self.bounds['alpha'] = {'start':0.01, 'stop':10.0, 'num':num0}  # prelec parameter
        self.bounds['delta'] = {'start':0.20, 'stop':0.80, 'num':num0}  # convexity
        self.bounds['theta'] = 1  # rationality
        self.keys = self.bounds.keys()
        self.CPT_support = {}
        self.params = {}
        self.preferences = np.array(0,dtype=object) # preferences for each model
        self.anomalies = np.array(0,dtype=object) # anomalies for each model
    def generate(self):
        self.CPT_support = {}
        for key in self.keys:
            bnds = self.bounds[key]
            self.CPT_support[key] = np.linspace(**bnds) if isinstance(bnds,dict) else [bnds]
        self.models = np.array(list(cartesian_product(*[self.CPT_support[key] for key in self.keys])))  # CPT models
        self.nModels = np.shape(self.models)[0]
        self.__getitem__(0) # init current model params
        self.preferences = np.array(self.nModels,dtype=object)
        self.anomalies = np.array(self.nModels, dtype=object)
    def summary(self):
        print(f'== CPT Model Initialization ==')
        print(f'\t| Models Size = {np.shape(self.models)}')
        print(f'\t| Support:')
        for key in self.keys:
            print(f'\t\t- {key}: {self.CPT_support[key]}')
    def preview(self):
        print(f'\n===== Current CPT Model =====')
        for key in self.params: print(f'\t| {key}: {self.params[key]}')
        if len(self.params.keys())==0: print(f'\t| [None]')
    def __getitem__(self, item):
        self.params = {}
        for iparam, key in enumerate(self.keys):
            self.params[key] = self.models[item,iparam]
        return self.params




if __name__ == "__main__":
    Mcpt = CPT_Models(num0=3)
    Mcpt.generate()
    # for iM in range(2):
    #     CPT_params = Mcpt[iM]
    #     Mcpt.preview()

    bandits = uniform_kArmedBandits()
    bandits.preview_game()
    bandits.evaluate()

    # TEST SINGLE STATS -------------
    # iM = 0
    # opt_pref = bandits.optimal_preferences()
    # cpt_pref = bandits.CPT_preferences(Mcpt[iM])
    # anomaly = bandits.preference_anomaly(Mcpt[iM])
    # print(f'optimal preferences = {opt_pref}')
    # print(f'CPT preferences     = {np.round(cpt_pref, 2)}')
    # print(f'preference anomaly  = {np.round(anomaly, 2)}')

    # Load all model stats -------------
    # for iM in range(Mcpt.nModels):
    #     cpt_pref = bandits.CPT_preferences(Mcpt[iM])
    #     anomaly = bandits.preference_anomaly(Mcpt[iM])
    #     Mcpt.preferences[iM] = cpt_pref
    #     Mcpt.anomalies[iM]   = anomaly


    # PLOT PREFERENCE COMPARISONS --------
    iM = 0
    opt_pref = bandits.optimal_preferences()
    cpt_pref = bandits.CPT_preferences(Mcpt[iM])
    anomaly = bandits.preference_anomaly(Mcpt[iM])
    print(f'optimal preferences = {opt_pref}')
    print(f'CPT preferences     = {np.round(cpt_pref, 2)}')
    print(f'preference anomaly  = {np.round(anomaly, 2)}')
    fig,axs = plt.subplots(2)
    bandits.plot_preferences(axs[0],anomaly,is_anomaly=True)
    bandits.plot_preferences(axs[1], cpt_pref, is_anomaly=False)
    fig.tight_layout()
    plt.show()
    Mcpt.preferences[iM] = bandits.CPT_preferences(Mcpt[0])

