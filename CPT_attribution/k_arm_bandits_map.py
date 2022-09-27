import copy
import math

import numpy as np
import pandas
from agents.policies import noisy_rational,CPT_Handler
from functools import partial
from itertools import product as cartesian_product
import matplotlib.pyplot as plt
# pandas.set_option('display.max_columns', None)
from tqdm import tqdm
def print_df_full(x,rows=True,cols=True):
    if rows: pandas.set_option('display.max_rows', None)
    if cols: pandas.set_option('display.max_columns', None)
    if cols:pandas.set_option('display.width', 2000)
    if cols:pandas.set_option('display.float_format', '{:20,.2f}'.format)
    if cols:pandas.set_option('display.max_colwidth', None)
    print(x)
    pandas.reset_option('display.max_rows')
    pandas.reset_option('display.max_columns')
    pandas.reset_option('display.width')
    pandas.reset_option('display.float_format')
    pandas.reset_option('display.max_colwidth')
class uniform_kArmedBandits(object):
    def __init__(self,p_split=0.5,r_dist=10,Ecent=1,case_res=50,rationality=1,verbose=False):
        """
        :param nA: number of actions
        :param nO: number of outcomes
        """
        if verbose:
            print(f'\n\n')
            print(f'###########################################################')
            print(f'################# UNIFORM k-Armed Bandits #################')
            print(f'###########################################################')
        # Define indexes ============================================
        self.iP = 0     # prob index
        self.iR = 1     # reward index
        self.iE = 2
        self.iOp = 0    # index for "positive" outcome fo ai: o^{+}
        self.iOn = 1    # index for "negative" outcome fo ai: o^{-}

        # Define static params ============================================
        self.p0 = p_split               # baseline probability to modify to enforce indifference
        self.r0 = r_dist              # baseline reward (+/-) to enforce indifference
        self.O = [self.iOp, self.iOn]   # outcomes of each action (pos or neg)
        self.Info = [self.iP,self.iR]   # information given for each action
        self.case_res = case_res

        # define game ============================================
        self.nA = 0                                 # number of initial actions
        self.nO = len(self.O)                       # number of outcomes
        self.nInfo = len(self.Info)                 # number of information vars for each action
        self.Eai = []                               # expected value for each action
        self.G = np.zeros([0,self.nO,self.nInfo])   # the game form

        # Populate game ============================================
        r_range = (0, r_dist, case_res)
        p_range = (-(p_split-0.01), p_split-0.01, case_res)
        E_range = (-Ecent, Ecent, 3) # E_range = self.E0
        self._Ehats = [E_range] if np.size(E_range)==1 else np.linspace(E_range[0],E_range[1],int(E_range[2]))
        self._phats = [p_range] if np.size(p_range)==1 else np.linspace(p_range[0],p_range[1],int(p_range[2]))
        self._rhats = [r_range] if np.size(r_range) == 1 else np.linspace(r_range[0], r_range[1], int(r_range[2]))

        # Define parameter space for game ===========================
        self.nCases = 0                         # number of test cases
        self.anomaly_cases = np.ndarray         # sequential list of test case params
        self.anomaly_icases = np.ndarray        # sequential list of test case indicies
        self.anomalymap = np.ndarray            # 3D matrix of preference anomalies w. iparam as indices
        self.anomalymap_params = np.ndarray     # 3D matrix of param values w. iparam as indices
        self.gambles = np.ndarray               # array of gambles to compare against baseline
        # self.gamble_baseline = self.get_gamble(self.p0,0,self.E0) # reference for comparing gamble preferences
        self.gamble_baseline = [self.get_gamble(self.p0, 0, E) for E in self._Ehats]  # reference for comparing gamble preferences
        self.generate_test_cases()              # fill above vars

        # Define decision models ===========================
        self.opt_rationality = rationality
        self.boltzmann_policy = noisy_rational  # optimal agent boltzmann policy
        # self.boltzmann_policy = partial(noisy_rational, rationality=1)  # optimal agent boltzmann policy
        pass


    #######################################################
    # GAME TOOLS ##########################################
    def generate_test_cases(self):
        ranges = [list, list, list]
        ranges[self.iP] = self._phats
        ranges[self.iR] = self._rhats
        ranges[self.iE] = self._Ehats
        sz_ranges = [np.size(tmp_range) for tmp_range in ranges]
        n_params = len(sz_ranges)

        # Define sequential list of test cases
        self.anomaly_cases = np.array(list(cartesian_product(*[ranges[i] for i in range(len(sz_ranges))])))  # np.empty(sz_ranges + [n_params])
        self.anomaly_icases = np.array(list(cartesian_product(*[np.arange(sz) for sz in sz_ranges])),dtype='int8')
        self.nCases = np.shape(self.anomaly_cases)[0]
        self.gambles = np.empty([self.nCases, self.nO,self.nInfo])

        # Define 3D array for test param values and anomaly result
        self.anomalymap = np.empty(sz_ranges)
        self.anomalymap_params = np.empty(sz_ranges + [n_params])
        for i, case in enumerate(self.anomaly_icases):
            params = np.copy(self.anomaly_cases[i])
            self.anomalymap_params[case] = params
            self.gambles[i] = self.get_gamble(params[self.iP], params[self.iR], params[self.iE])
            # self.anomalymap[icase] = preference_anomaly # result

    def get_gamble(self,phat,rhat,Ehat):
        """
        :param phat: modified probability
        :param rhat: magnatude of pos and neg reward | Exp Value = 0
        :param Ehat: bias on expected value to make expected reward pos/neg
        :return:
        """
        psym = self.p0                          # probability given symmetric gamble
        pp = psym + phat                        # prob of negative outcome
        pn = psym - phat                        # prob of positive outcome
        pos_rscale = ((psym - phat)/psym)       # scale positive reward given modified prob with phat
        neg_rscale = ((psym + phat) / psym)     # scale negative reward given modified prob with phat
        gi = np.zeros([1, self.nO,self.nInfo])  # temp game line

        if pp == 1: # certain positive outcome
            gi[0, self.iOp, [self.iP,self.iR]] = [pp, pos_rscale * rhat + Ehat]
            gi[0, self.iOn, :] = np.NAN
        elif pn == 1:  # certain negative outcome
            gi[0, self.iOp, :] = np.NAN
            gi[0, self.iOn, [self.iP,self.iR]] = [pn, neg_rscale*(-1*rhat) + Ehat]
        else: # gamble outcome
            gi[0, self.iOp, [self.iP, self.iR]] = [pp, pos_rscale * rhat + Ehat]
            gi[0, self.iOn, [self.iP, self.iR]] = [pn, neg_rscale * (-1 * rhat) + Ehat]

            # #"positive" outcome fo ai: o^{+}
            # gi[0,self.iOp,self.iP] = pp
            # gi[0,self.iOp,self.iR] = pos_rscale* rhat + Ehat
            #
            # # "negative" outcome fo ai: o^{-}
            # gi[0, self.iOn, self.iP] = pn
            # gi[0, self.iOn, self.iR] = neg_rscale*(-1*rhat) + Ehat

        # return gamble info
        if np.any(gi[:,:,self.iP]<0):
            raise Exception(f'Negative probabilities in gamble p = {gi[:,:,self.iP]}')
        return gi
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
        if len(np.shape(G)) == 2: G = G.reshape([1]+list(np.shape(G)))
        # [ai, iO,:]
        isnans = [np.any([math.isnan(G[0,io,self.iP]), math.isnan(G[0,io,self.iR]) ]) for io in range(np.shape(G)[1])]



        if np.any(isnans):
            Eai = np.prod(G[:,[not val for val in isnans],:],axis = 2).flatten()[0]
        else: Eai = np.sum(np.prod(G[:,:,:],axis = 2),axis=1)
        return Eai

    #######################################################
    # POLICY PREFERENCES ##################################
    def get_attitude(self, anomaly_mat, hThresh = 0.25, lThresh = 0.1 ):
        if np.mean(anomaly_mat) > hThresh: attitude = 'HIGH Risk Sensitivity'  # seeks certainty
        elif np.mean(anomaly_mat) > lThresh:  attitude = 'LOW Risk Sensitivity'  # prefers certainty
        elif np.mean(anomaly_mat) >= 0: attitude = 'Risk-Insensitive'  # close enough to noisy
        elif np.mean(anomaly_mat) < 0: attitude = 'Risk SEEKING'  # prefers taking risks
        else: attitude = 'MIXED Risk Behaviors'  # depends on context
        return attitude
    def get_preference_anomaly_map(self,CPT_params,icomp=1,assume_opt=False,progbar = False,verbose=False):
        getData = True
        if verbose:
            sigdig= 2
            param_list = [f'{key}:{CPT_params[key]}' for key in CPT_params]
            print(f'\nGenerating anomaly map...')
            print(f'\t| {param_list}')
            col_space = '  '
            col_fill = '|'
            if getData: cols = ['pmod','rdist','Ecent',col_space,'OPT_pref','OPT_E',col_space,'CPT_pref','CPT_E','Anomaly']
            else: cols = ['pmod','rdist','Ecent',col_space,'OPT+','OPT-',col_space,'CPT+','CPT-']
            data = np.zeros([self.nCases, len(cols)],dtype=object)
            iempty = np.where(np.array([1 if cols[c] == col_space else 0 for c in range(len(cols))])==1)[0]
            iopt = np.where(np.array([1 if 'OPT' in cols[c] else 0 for c in range(len(cols))])==1)[0]
            icpt = np.where(np.array([1 if 'CPT' in cols[c] else 0 for c in range(len(cols))])==1)[0]
            data[:,iempty] = col_fill

        for icase in tqdm(range(self.nCases)) if progbar else range(self.nCases):
            popt, Eopt = self.optimal_preferences(icase, getData=getData)
            pcpt, Ecpt = self.CPT_preferences(CPT_params, icase, getData=getData)
            anomaly = (popt-pcpt)[icomp] # result
            #print(self.anomaly_icases[icase],np.round(anomaly,2))

            params = self.anomaly_cases[icase]

            ip = np.where(self._phats == params[self.iP])[0][0]
            ir = np.where(self._rhats == params[self.iR])[0][0]
            iE = np.where(self._Ehats == params[self.iE])[0][0]

            self.anomalymap[ir,ip,iE] = anomaly # <=========    CHANGED ORDER OF VARS =========

            # self.anomalymap[self.anomaly_icases[icase]] = anomaly

            if verbose:
                data[icase, 0:3] = self.anomaly_cases[icase]
                data[icase, iopt[0]] = np.round([popt[0], popt[1]], sigdig)
                data[icase, iopt[1]] = np.round(Eopt, sigdig)
                data[icase, icpt[0]] = np.round([pcpt[0], pcpt[1]], sigdig)
                data[icase, icpt[1]] = np.round(Ecpt, sigdig)
                data[icase, -1] = anomaly

            # print(f'opt: {np.round(popt,1)} \t cpt: {np.round(pcpt,1)} \t anomaly: {np.round(popt-pcpt,1)}')

        if verbose:
            df = pandas.DataFrame(data, columns=cols)
            print_df_full(df, rows=False)


        return self.anomalymap
    def plot_heatmap2D(self,ax,A,labels=None,cmap='RdBu'):
        """  :param cmap:  coolwarm RdBu  """
        im = ax.imshow(A,cmap=cmap,vmin=-0.6,vmax=0.6,)
        nticks=5
        nP, nR = A.shape

        y_labels = [tick for tick in np.linspace(min(self._rhats), max(self._rhats), nticks)]
        y_locs = np.linspace(0, nP, nticks)
        ax.set_yticks(y_locs)
        ax.set_yticklabels(y_labels)

        x_labels = [tick - min(self._phats) for tick in np.linspace(min(self._phats),max(self._phats),nticks)]
        x_locs = np.linspace(0,nR,nticks)
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_labels)



        ax.set_aspect('equal', adjustable='box')
        if labels is not None:
            if 'title' in labels.keys(): ax.set_title(labels['title'])
            if 'x' in labels.keys(): ax.set_xlabel(labels['y'])
            if 'y' in labels.keys():  ax.set_ylabel(labels['x'])
            if 'colorbar' in labels.keys(): pass
        return im
    def plot_transforms(self,axR,axP,CPT_params,res=100):
        c_CPT = 'orange'
        c_OPT = 'black'
        c_GUIDE= 'lightgrey'

        CPT_policy = CPT_Handler(**CPT_params)

        r = np.linspace(-self.r0,self.r0,res)
        rperc = CPT_policy.utility_weight(r)
        axR.plot(r, r, label='Optimal',color=c_OPT)
        axR.plot(r,rperc,label='CPT',color=c_CPT)
        axR.set_xlabel('Rel. Reward (r-b)')
        axR.set_ylabel('Perceived Reward')
        axR.set_title('Reward Transform')
        axR.hlines(0 ,xmin=min(r),xmax=max(r),ls=':',color=c_GUIDE)
        axR.vlines(0, ymin=min(r), ymax=max(r), ls=':', color=c_GUIDE)
        axR.set_xlim([min(r),max(r)])
        axR.set_ylim([min(r), max(r)])
        axR.set_aspect('equal', adjustable='box')
        axR.legend()

        p = np.linspace(0, 1, res)
        pperc = CPT_policy.prob_weight(p)
        axP.plot(p, p, label='Optimal', color=c_OPT)
        axP.plot(p, pperc, label='CPT', color=c_CPT)
        axP.set_xlabel('Probability (p)')
        axP.set_ylabel('Perceived Prob.')
        axP.set_title('Probability Transform')
        axP.set_xlim([min(p), max(p)])
        axP.set_ylim([min(p), max(p)])
        axP.legend()
        axP.set_aspect('equal', adjustable='box')
    def optimal_preferences(self,igamble,getData=False):
        nA, nO = np.shape(self.gamble_baseline[0])[1:]
        iBL = self.anomaly_icases[igamble,self.iE] # baseline index

        G = np.append(self.gamble_baseline[iBL], self.gambles[igamble].reshape([1, nA, self.nO]), axis=0)  # game form
        Eai= np.empty(nA)
        for ai in range(nA): Eai[ai] = self.evaluate(G = G[ai]) # get expected value
        prefs = np.array( self.boltzmann_policy(Eai,rationality=self.opt_rationality))
        if getData: return prefs,Eai
        else: return prefs
    def CPT_preferences(self,CPT_params,igamble,getData=False,verbose = False):
        CPT_policy = CPT_Handler(**CPT_params)
        nA, nO = np.shape(self.gamble_baseline[0])[1:]
        iBL = self.anomaly_icases[igamble, self.iE]  # baseline index

        G = np.append(self.gamble_baseline[iBL],self.gambles[igamble].reshape([1,nA,self.nO]),axis=0)   # game form
        G_perc = np.empty(np.shape(G))
        Eai_perc = np.empty(nA)

        for ai, iO in cartesian_product(*[np.arange(nA), np.arange(nO)]):
            choice = G[ai,iO,:]
            # G_perc[ai, iO, self.iR] = CPT_policy.utility_weight(choice[self.iR])
            # G_perc[ai, iO, self.iP] = CPT_policy.prob_weight(choice[self.iP])


            if np.any([math.isnan(c) for c in choice]):
                G_perc[ai,iO,:] = choice
            else:
                G_perc[ai,iO,self.iR] = CPT_policy.utility_weight(choice[self.iR])
                G_perc[ai,iO,self.iP] = CPT_policy.prob_weight(choice[self.iP])
            if verbose: print(f'{np.round(choice,2)} ==> {np.round(G_perc[ai,iO,:],2)}')
            pass

        for ai in range(nA):
            Eai_perc[ai] = self.evaluate(G = G_perc[ai]) # get expected value
        prefs = np.array(CPT_policy.boltzmann(Eai_perc))
        if verbose:  print(f'pref = {np.round(prefs,2)} E = {np.round(Eai_perc,1)}')
        if getData: return prefs,Eai_perc
        else: return prefs


class CPT_Models(object):
    def __init__(self,num0 = 10):
        """
        !!!!!!!!!!!!!! NEEED TO ADJUST EQUAL NUMBER OF VALS BELOW AND ABOVE 1 for ALPHA AND OTHERS !!!!!!!
        """
        # Define Feasable Bounds
        """ Feasible bounds for loss aversion technically is >1"""
        self.BOUNDS01 = [0.2,0.8]
        self.BOUNDS0inf = [0.1,10]


        self.def_bounds = {}
        self.def_bounds['b']     = 0.0                                  # reference point
        self.def_bounds['gamma'] = dict(start=0.2, stop=0.8, num=num0)  # diminishing (sensitivity) return gain
        self.def_bounds['lam']   = dict(start=1.0, stop=8.0, num=num0)  # loss aversion
        self.def_bounds['alpha'] = dict(start=0.5, stop=5, num=num0)  # prelec parameter
        self.def_bounds['delta'] = dict(start=0.2, stop=0.8, num=num0)  # probability sensitivity
        self.def_bounds['theta'] = dict(start=1.0, stop=10.0, num=num0)  # rationality

        # Create Working Bounds and Fix parameters -------------
        self.bounds = {}
        self.bounds = copy.deepcopy(self.def_bounds)
        self.bounds['b'] = 0 # reference point
        # self.bounds['alpha'] = 1  # prelec parameter
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
            if isinstance(bnds, dict):
                if key == 'alpha': # do equal parts above and below 1
                    nadj = round(bnds['num']/2)
                    # vals0 = np.linspace(bnds['start'], 1-bnds['start']/nadj, nadj)
                    vals0 = 1/np.linspace(1+(1/bnds['stop'])/nadj, (1/bnds['start']), nadj)
                    vals1 = np.linspace(1+bnds['stop']/nadj, bnds['stop'],nadj)
                    # vals0 = 1/vals1
                    self.CPT_support[key] = np.append(vals0,vals1)
                else:
                    self.CPT_support[key] = np.linspace(bnds['start'],bnds['stop'],bnds['num'])
            else: self.CPT_support[key] = [bnds]
            # self.CPT_support[key] = np.linspace(**bnds) if isinstance(bnds,dict) else [bnds]
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
    # np.random.seed(1)
    np.random.seed(2) # good
    np.random.seed(5)
    Mcpt = CPT_Models()
    Mcpt.generate()


    hSensitivity = 0.25
    lSensitivity = 0.1
    PLOT_ATTITUDE = True
    if PLOT_ATTITUDE:
        # CPT_Model = Mcpt.examples['optimal']
        # CPT_Model = Mcpt.examples['loss_ignorant']
        # CPT_Model = Mcpt.examples['loss_averse']
        # CPT_Model = Mcpt.examples['hyper_reward_discounting']
        # CPT_Model = Mcpt.examples['hyper_prob_discounting']
        # CPT_Model = Mcpt.examples['overestimate_prob']
        # CPT_Model = Mcpt.examples['underestimate_prob']
        # CPT_Model = Mcpt[iM]
        # np.random.seed(0)


        bandits = uniform_kArmedBandits(case_res=100)
        CPT_Model = Mcpt[np.random.choice(np.arange(Mcpt.nModels))]
        Mcpt.preview()
        anomalymap = bandits.get_preference_anomaly_map(CPT_Model)
        nE = np.shape(anomalymap)[2]

        fig, axs = plt.subplots(2, 3)
        bandits.plot_transforms(axs[0, 0], axs[0, 1], CPT_Model)

        if nE==1: Elabels = [' ($E(A)=0$)']
        else: Elabels = [' ($E(A)<0$)',' ($E(A)=0$)',' ($E(A)>0$)']
        for iE0 in range(nE):
            attitude = bandits.get_attitude(anomalymap[:,:,iE0],hThresh=hSensitivity,lThresh=lSensitivity)
            ax_labels = {'title': f'Preference Anomaly {Elabels[iE0]} \n{attitude}', 'x': '$r_{dist}$', 'y': '$p_{mod}$'}
            im = bandits.plot_heatmap2D(axs[1,iE0],anomalymap[:,:,iE0],labels=ax_labels)
        clb = fig.colorbar(im)
        clb.ax.set_title('Certainty\n Preference')
        fig.tight_layout()
        print(f'General Attitude: {bandits.get_attitude(anomalymap, hThresh=hSensitivity, lThresh=lSensitivity)}')







    # FIND LOW RISK SENSITIVITY
    GET_ATTITUDES = False
    if GET_ATTITUDES:
        bandits = uniform_kArmedBandits()
        attitude_list = []
        for iM in [0,1,2]:#range(Mcpt.nModels):
            CPT_Model = Mcpt[iM]
            anomalymap = bandits.get_preference_anomaly_map(CPT_Model)
            nE = np.shape(anomalymap)[2]
            bandits.plot_transforms(axs[0, 0], axs[0, 1], CPT_Model)
            print(f'\rModel {iM}:', end='')
            attitude = bandits.get_attitude(anomalymap,hThresh=hSensitivity,lThresh=lSensitivity)
            attitude_list.append(attitude)
        np.save('attitudes',np.array(attitude_list,dtype=object),allow_pickle=True)
        # loaded = np.load('attitudes.npy',allow_pickle=True)


    plt.show()