import copy
import math

import numpy as np
import pandas
from agents.policies import noisy_rational,CPT_Handler
from functools import partial
from itertools import product as cartesian_product
import matplotlib.pyplot as plt
# pandas.set_option('display.max_columns', None)

def print_df_full(x):
    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.width', 2000)
    pandas.set_option('display.float_format', '{:20,.2f}'.format)
    pandas.set_option('display.max_colwidth', None)
    print(x)
    pandas.reset_option('display.max_rows')
    pandas.reset_option('display.max_columns')
    pandas.reset_option('display.width')
    pandas.reset_option('display.float_format')
    pandas.reset_option('display.max_colwidth')
class uniform_kArmedBandits(object):
    def __init__(self):
        """
        :param nA: number of actions
        :param nO: number of outcomes
        """
        print(f'\n\n')
        print(f'###########################################################')
        print(f'################# UNIFORM k-Armed Bandits #################')
        print(f'###########################################################')
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
        self._phats['a_cert'] = 0.5
        self._phats['a_sym'] = 0
        self._phats['a_p+'] = 0.25
        self._phats['a_p-'] = -0.25
        # self._phats['p=1,r=0'] = 0.5
        # self._phats['$(p,r)_+=(0.5,10)$\n$(p,r)_-=(0.5,-10)$'] = 0
        # self._phats['$(p,r)_+=(0.75,5)$\n$(p,r)_-=(0.25,-15)$'] = 0.25
        # self._phats['$(p,r)_+=(0.25,15)$\n$(p,r)_-=(0.75,-5)$'] = -0.25
        for ai, key in enumerate(self._phats.keys()):  self.add_gamble(phat=self._phats[key])

        # Comparison Modes
        self.COMPARISON_MODES = ['population','binary']
        self._cmp_mode = self.COMPARISON_MODES[1]

        self._icmp0 = 0                                     # compare against action index
        self._icmp0_def = np.NAN                            # default value for compared action
        self._cmps = list(range(self.nA)).pop(self._icmp0)  # comparison indexes
        # self._binary_comparisons = [[self._icmp0,ai] for ai in range(self.nA-1)] # index comparison handlers
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
    def preview_game(self,G = None, notes=''):

        print(f'\n\n============================')
        print(f'Game Preview: {notes}')
        print(f'============================')
        if G is None: G = self.G
        data_dict = {}
        col_headers = ['Eai',' ','p+', 'r+','[V+]',' ','p-', 'r-','[V+]']
        sig_dig = 2
        for ai, key in enumerate(self._phats.keys()):
            pinfo = G[ai, self.iOp, :]             # positive info
            ninfo = G[ai, self.iOn, :]             # negative info
            Ep = np.product(G[ai, self.iOp, :])    # positive expected reward
            En = np.product(G[ai, self.iOn, :])    # negative expected reward
            Eai = Ep+En
            data_dict[key] = [Eai, ' ',pinfo[self.iP], pinfo[self.iR],Ep, ' ',ninfo[self.iP], ninfo[self.iR],En]
        df = pandas.DataFrame.from_dict(data_dict, orient='index', columns=col_headers)
        # print(df.round(sig_dig))
        print(df)
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
            # prefs = [self._icmp0_def] + [self.boltzmann_policy(Eai[icmp],rationality=rat)[1] for icmp in self._binary_comparisons]
            prefs = [self._icmp0_def] + [self.boltzmann_policy([self.E0, Eai[ai]], rationality=rat)[1] for ai in range(self.nA)]
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
        self.preview_game(G =G_perc,notes='PERCEIVED')
        Eai_perc = self.evaluate(G=G_perc)

        if self._cmp_mode == 'binary':
            # prefs = [self._icmp0_def] + [CPT_policy.boltzmann(Eai_perc[icmp])[1] for icmp in self._binary_comparisons]
            prefs = [CPT_policy.boltzmann([self.E0,Eai_perc[ai]])[1] for ai in range(self.nA)]
        elif self._cmp_mode == 'population':  prefs = CPT_policy.boltzmann(Eai_perc)
        else: raise Exception('in uniform_kArmedBandits >> unknown comparison mode')
        return np.array(prefs)
    def intepret_preferences(self,pref):
        anomaly = pref - self.uniform_preference
        labels = list(self._phats.keys())
        descr = {}

        descr[labels[0]] = 'a certain outcome'
        descr[labels[1]] = 'symm inv outcomes'
        descr[labels[2]] = 'a high P(r+) outcome'
        descr[labels[3]] = 'a high P(r-) outcome'

        print(f'\n\n')
        print(f'###################################################')
        print(f'Interpretation for a {self._cmp_mode} Comparison')
        print(f'###################################################')
        sig_thresh = 0.2

        # cols = ['between', 'A1', 'A2','','magnitude','effect','on']
        # cols = ['between', 'a_0', 'a_pc','there is a','significant','increase','in preference for a_pc']
        data = []
        if self._cmp_mode == 'binary':
            for ai in np.arange(len(labels)):
                A1 = 'Certain R=0 '
                A2 = labels[ai]
                anm = anomaly[ai]
                if anm == 0: mag,effect = 'no','change'
                else:
                    mag = 'significant' if np.abs(anm) > sig_thresh else 'small'
                    effect = 'increase' if anm >0 else 'decrease'
                data_line = ['between', A1, A2, 'there is a',mag,effect, f'in preference for {A2}']
                data.append(data_line)
        data = np.array(data)



        print_df_full(pandas.DataFrame(data))
    def plot_preferences(self,ax,preference,anomaly=None,title = 'Preference',notes=''):
        c = ['dimgrey','orange','maroon']
        labels =list(self._phats.keys())
        unf_pref = self.uniform_preference
        txt_fmt = '%.2f'
        eps = 4e-3

        width = 0.35 if anomaly is None else 0.3  # the width of the bars
        pad = 3

        if self._cmp_mode == 'binary':
            # labels = labels[1:]
            # anomaly = anomaly[1:]
            # preference = preference[1:]
            # unf_pref = unf_pref[1:]
            title += '(Binary Comparison with Certain R=0)'
        elif self._cmp_mode == 'population': title += '(Comparison Across all Actions)'
        else: raise Exception('unknown comparison mode')
        x = np.arange(len(labels))  # the label locations
        title += f': <{notes}>'


        preference[np.where(preference == 0)] = eps


        # plot against uniform preference
        if anomaly is None:
            rects_opt = ax.bar(x - width / 2, unf_pref, width, label='Optimal',color = c[0])
            rects_cpt = ax.bar(x  + width / 2, preference, width, label='CPT',color = c[1])
            ax.bar_label(rects_opt, padding=pad,fmt=txt_fmt)
            ax.bar_label(rects_cpt, padding=pad,fmt=txt_fmt)

        else:
            anomaly[np.where(anomaly == 0)] = eps
            rects_opt = ax.bar(x - width, unf_pref,     width, label='Optimal',color = c[0])
            rects_cpt = ax.bar(x        , preference,   width, label='CPT',color = c[1])
            rects_anm = ax.bar(x + width, anomaly,      width, label='Anomaly',color = c[2])
            ax.bar_label(rects_opt, padding=pad,fmt=txt_fmt)
            ax.bar_label(rects_cpt, padding=pad,fmt=txt_fmt)
            ax.bar_label(rects_anm, padding=pad,fmt=txt_fmt)


        ax.set_title(title + '\n')
        ax.set_ylabel('P(A| E{A})')
        ax.set_xticks(x, [f'{lbl}' for lbl in labels])
        ax.legend()


class CPT_Models(object):
    def __init__(self,num0 = 3):
        """
        !!!!!!!!!!!!!! NEEED TO ADJUST EQUAL NUMBER OF VALS BELOW AND ABOVE 1 for ALPHA AND OTHERS !!!!!!!

        :param num0:
        """
        # Define Feasable Bounds
        """ Feasible bounds for loss aversion technically is >1"""
        self.BOUNDS01 = [0.2,0.8]
        self.BOUNDS0inf = [0.1,10]


        self.def_bounds = {}
        self.def_bounds['b']     = 0.0                                  # reference point
        self.def_bounds['gamma'] = dict(start=0.2, stop=0.8, num=num0)  # diminishing (sensitivity) return gain
        self.def_bounds['lam']   = dict(start=0.1, stop=10.0, num=num0)  # loss aversion
        self.def_bounds['alpha'] = dict(start=0.1, stop=10, num=num0)  # prelec parameter
        self.def_bounds['delta'] = dict(start=0.2, stop=0.8, num=num0)  # probability sensitivity
        self.def_bounds['theta'] = dict(start=0.1, stop=10.0, num=num0)  # rationality

        # Create Working Bounds and Fix parameters -------------
        self.bounds = {}
        self.bounds = copy.deepcopy(self.def_bounds)
        self.bounds['b'] = 0 # reference point
        self.bounds['alpha'] = 1  # prelec parameter
        self.bounds['theta'] = 1  # rationality



        self.keys = self.bounds.keys()
        self.CPT_support = {}
        self.params = {}
        self.preferences = np.array(0,dtype=object) # preferences for each model
        self.anomalies = np.array(0,dtype=object) # anomalies for each model


        # Example attributions by modify default (optimal) params
        self.CPT_def = {}
        self.CPT_def['b']       = 0     # reference point
        self.CPT_def['gamma']   = 1     # diminishing return gain
        self.CPT_def['lam']     = 1     # loss aversion
        self.CPT_def['alpha']   = 1     # prelec parameter
        self.CPT_def['delta']   = 1     # Probability weight
        self.CPT_def['theta']   = 1     # rationality
        self.examples = self.generate_exampels()
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
    def generate_exampels(self):
        examples = {}
        examples['optimal'] = copy.deepcopy(self.CPT_def)

        # Also known as los aversion
        param = 'lam'
        examples['loss_ignorant'] = copy.deepcopy(self.CPT_def)
        examples['loss_ignorant'][param] = self.def_bounds[param]['start']  # low val
        examples['loss_averse'] = copy.deepcopy(self.CPT_def)
        examples['loss_averse'][param] = self.def_bounds[param]['stop']  # high val

        # Discounting of diminishing returns
        param = 'gamma'
        examples['hyper_reward_discounting'] = copy.deepcopy(self.CPT_def)
        examples['hyper_reward_discounting'][param] = self.def_bounds[param]['start']  # low val
        # examples['hypo_reward_discounting'] = copy.deepcopy(self.CPT_def)
        # examples['hypo_reward_discounting'][param] = self.def_bounds[param]['stop']  # high val

        # Prelec parameter
        param = 'alpha'
        examples['overestimate_prob'] = copy.deepcopy(self.CPT_def)
        examples['overestimate_prob'][param] = self.def_bounds[param]['start']  # low val
        examples['underestimate_prob'] = copy.deepcopy(self.CPT_def)
        examples['underestimate_prob'][param] = self.def_bounds[param]['stop']  # high val

        # Probability weight
        param = 'delta'
        examples['hyper_prob_discounting'] = copy.deepcopy(self.CPT_def)
        examples['hyper_prob_discounting'][param] = self.def_bounds[param]['start']  # low val
        # examples['hypo_reward_discounting'] = copy.deepcopy(self.CPT_def)
        # examples['hypo_reward_discounting'][param] = self.def_bounds[param]['stop']  # high val
        return examples
    def summary(self,support = None):
        if support is None: support = self.CPT_support
        print(f'\n\n')
        print(f'== CPT PARAMS ==')
        print(f'\t| Models Size = {np.shape(self.models)}')
        # print(f'\t| Support:')
        cols = list(self.keys)
        data = {'params': [support[key] for key in self.keys]}
        print(pandas.DataFrame.from_dict(data,orient='index',columns=cols))
        # for key in self.keys:
        #     print(f'\t\t- {key}: {support[key]}')
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


    # EVALUATE EXAMPLES --------
    # iM = 0
    # CPT_params = Mcpt[iM]
    # CPT_params = Mcpt.examples['optimal']
    # CPT_params = Mcpt.examples['loss_averse']
    # CPT_params = Mcpt.examples['hyper_reward_discounting']
    # CPT_params = Mcpt.examples['overestimate_prob']
    # CPT_params = Mcpt.examples['underestimate_prob']
    # CPT_params = Mcpt.examples['hyper_prob_discounting']
    attributions = ['optimal', None,
                    'loss_ignorant', 'loss_averse',
                    'hyper_reward_discounting', 'hyper_prob_discounting',
                    'overestimate_prob', 'underestimate_prob']

    fig, axs = plt.subplots(math.ceil(len(attributions)/2),2)
    r,c = 0,0
    for iax, attr in enumerate(attributions):
        if attr is None:
            c+=1
        else:
            CPT_params = Mcpt.examples[attr]

            # Mcpt.summary(support=CPT_params)
            # opt_pref = bandits.optimal_preferences()
            cpt_pref = bandits.CPT_preferences(CPT_params)
            # anomaly = bandits.preference_anomaly(CPT_params)
            bandits.intepret_preferences(cpt_pref)

            # PLOT PREFERENCE COMPARISONS --------
            try:
                # bandits.plot_preferences(axs,cpt_pref,anomaly=anomaly)
                bandits.plot_preferences(axs[r,c], cpt_pref,notes=attr)
                c +=1
            except:
                r,c = r+1,0
                # bandits.plot_preferences(axs,cpt_pref,anomaly=anomaly)
                bandits.plot_preferences(axs[r,c], cpt_pref,notes=attr)
                c +=1

        # fig.tight_layout()
    plt.show()
        # Mcpt.preferences[iM] = bandits.CPT_preferences(Mcpt[0])

