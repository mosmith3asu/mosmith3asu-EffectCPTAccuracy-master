import copy
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas
from matplotlib.widgets import Slider, Button, RadioButtons,TextBox,AxesWidget
from agents.policies import noisy_rational,CPT_Handler
from functools import partial
import matplotlib.gridspec as gridspec
from k_arm_bandits_map import uniform_kArmedBandits
from random import randrange
from scipy.optimize import minimize


class PlotWidgets(object):
    def __init__(self,loc='right',size = 0.3,pad = (0.05,0.05)):


        self.pad = pad
        self.loc = loc
        self.widget_box_sz = size
        if loc == 'left':
            self.widget_box = [0,None,size,1]
            # plt.subplots_adjust(left=size,right=self.pad[0])
        elif loc == 'right':
            self.widget_box = [1-size+self.pad[1],None,size-pad[1],1]
            # plt.subplots_adjust(left=self.pad[0],right=1-size)
        self.y0 = 0.9  # current cumulative yloc of widgetrs

        self.farthest_xloc = 1 if loc == 'right' else 0
        self.widgets = {}


    def adj_subplots(self,widget):
        label = widget.ax.get_children()[1]
        self.farthest_xloc = min(abs(self.farthest_xloc), np.min(label.xy[:, 0]))


    def adj_label(self, widget, adj = 0.1):
        pos = list(widget.ax.get_position().bounds)
        pos[0] += adj
        pos[2] -= adj
        widget.ax.set_position(pos)

    def add_slider(self,label,vrange,ref=None, vinit=None, height=0.03,nstep= 50,ax=None,step=None):
        step = (vrange[1]-vrange[0])/nstep if step is None else step
        vinit = np.mean(vrange) if vinit is None else vinit
        # if ax is None: ax = plt.axes([self.widget_box[0], self.y0, self.widget_box[2]-self.pad[1], height])
        slider = Slider(ax, label, vrange[0], vrange[1], valinit=vinit, valstep=step)
        ref = label if ref is None else ref
        self.widgets[ref] = slider
        self.adj_subplots(slider)
        self.adj_label(slider)
        self.y0 -= height
        return slider

    def add_button(self,label,ref=None,height=0.03,ax=None):
        # if ax is None: ax = plt.axes([self.widget_box[0], self.y0, self.widget_box[2]-self.pad[1], height])
        button = Button(ax, label) #color=axcolor, hovercolor='0.975'
        ref = label if ref is None else ref
        self.widgets[ref] = button
        self.y0 -= height
        return button

    def add_radio(self,label,options,ref=None,vinit=0,height=0.03,ax=None):
        self.y0 -= height*(len(options)-0.5)
        # if ax is None: ax = plt.axes([self.widget_box[0], self.y0, self.widget_box[2]-self.pad[1], height*len(options)])#, facecolor=axcolor
        radio = RadioButtons(ax,options, active=vinit)
        ref = label if ref is None else ref
        self.widgets[ref] = radio
        self.y0 -= height
        return radio

    def add_textbox(self,label,ref=None,vinit='',height=0.03,ax=None):
        # if ax is None: ax = plt.axes([self.widget_box[0], self.y0, self.widget_box[2]-self.pad[1], height])#, facecolor=axcolor
        text_box = TextBox(ax,label, initial=vinit)
        ref = label if ref is None else ref
        self.widgets[ref] = text_box
        self.adj_subplots(text_box)
        self.y0 -= height
        return text_box

    def add_text(self, text, ref=None, pady =0.02, xadj = 0.0, height=0.03,bg = 'k',fg='w',align='left',ax=None):
        # if ax is None: ax = plt.axes([self.widget_box[0]-xadj, self.y0-pady, self.widget_box[2] - self.pad[1]+xadj, height])  # , facecolor=axcolor
        text_box = TextBox(ax, '', initial=text,color = bg, hovercolor=bg,textalignment=align,label_pad=-0)
        text_box.set_active = False
        text_box.text_disp.set_color(fg)  # text inside the edit box
        text_box.text_disp.set_fontweight('bold')
        # text_box.text_disp.set_verticalalignment('top')
        ref = text if ref is None else ref
        self.widgets[ref] = text_box
        self.y0 -=(height+pady)
        return text_box
    def vspace(self,height=0.03,ax=None):
        self.y0 -= height


def plot_transforms(axR,axP,CPT_params,rrange = (-10,10),res=50):
    global lCPT_r
    global lCPT_p
    c_CPT = 'orange'
    c_OPT = 'black'
    c_GUIDE= 'lightgrey'

    CPT_policy = CPT_Handler(**CPT_params)
    r = np.linspace(rrange[0],rrange[1],res)
    rperc = CPT_policy.utility_weight(r)
    axR.plot(r, r, label='Optimal',color=c_OPT)
    lCPT_r, = axR.plot(r,rperc,label='CPT',color=c_CPT)
    axR.set_xlabel('Rel. Reward (r-b)')
    axR.set_ylabel('Perceived Reward')
    axR.set_title('Reward Transform')
    axR.hlines(0 ,xmin=min(r),xmax=max(r),ls=':',color=c_GUIDE)
    axR.vlines(0, ymin=min(r), ymax=max(r), ls=':', color=c_GUIDE)
    axR.set_xlim([min(r),max(r)])
    axR.set_ylim([min(r), max(r)])
    # axR.set_aspect('equal', adjustable='box')
    # axR.set_aspect('equal')
    axR.legend()

    p = np.linspace(0, 1, res)
    pperc = CPT_policy.prob_weight(p)
    axP.plot(p, p, label='Optimal', color=c_OPT)
    lCPT_p, = axP.plot(p, pperc, label='CPT', color=c_CPT)
    axP.set_xlabel('Probability (p)')
    axP.set_ylabel('Perceived Prob.')
    axP.set_title('Probability Transform')
    axP.set_xlim([min(p), max(p)])
    axP.set_ylim([min(p), max(p)])
    axP.legend()
    # axP.set_aspect('equal', adjustable='box')
    # axP.set_aspect('equal')

def redraw(event,Widgets,rrange = (-10,10),res=50):
    global lCPT_r
    global lCPT_p
    # Widgets.widgets['redraw_button'].label.set_text('loading...')
    plt.draw()


    CPT_params = {}
    CPT_params['b'] = Widgets.widgets['b'].val  # reference point
    CPT_params['gamma'] = Widgets.widgets['gamma'].val   # diminishing return gain
    CPT_params['lam'] = Widgets.widgets['ell'].val   # loss aversion
    CPT_params['alpha'] = Widgets.widgets['alpha'].val   # prelec parameter
    CPT_params['delta'] = Widgets.widgets['delta'].val   # Probability weight
    CPT_params['theta'] = Widgets.widgets['lambda'].val   # rationality

    CPT_policy = CPT_Handler(**CPT_params)
    r = np.linspace(rrange[0], rrange[1], res)
    rperc = CPT_policy.utility_weight(r)
    p = np.linspace(0, 1, res)
    pperc = CPT_policy.prob_weight(p)
    lCPT_r.set_ydata(rperc)
    lCPT_p.set_ydata(pperc)
    # Widgets.widgets['redraw_button'].label.set_text('Redraw Anomalies')



# def plot_anomaly(axs,CPT_Model,p_split=0.5,r_dist=10,Ecent=1,case_res=50):
def plot_anomaly(axs, Widgets):
    global imgs

    settings = {}
    # settings['p_split'] = Widgets.widgets['phat'].val
    settings['r_dist'] = Widgets.widgets['rhat'].val
    settings['Ecent'] = Widgets.widgets['ehat'].val
    # settings['rationality'] = Widgets.widgets['lambda'].val
    settings['case_res'] = Widgets.widgets['resolution'].val

    CPT_Model = {}
    CPT_Model['b'] = Widgets.widgets['b'].val  # reference point
    CPT_Model['gamma'] = Widgets.widgets['gamma'].val  # diminishing return gain
    CPT_Model['lam'] = Widgets.widgets['ell'].val  # loss aversion
    CPT_Model['alpha'] = Widgets.widgets['alpha'].val  # prelec parameter
    CPT_Model['delta'] = Widgets.widgets['delta'].val  # Probability weight
    CPT_Model['theta'] = Widgets.widgets['lambda'].val  # rationality

    # global clb
    hSensitivity = 0.25
    lSensitivity= 0.1

    # bandits = uniform_kArmedBandits(p_split=p_split,r_dist=r_dist,Ecent=Ecent,case_res=case_res)
    bandits = uniform_kArmedBandits(r_dist=settings['r_dist'] ,
                                    Ecent=settings['Ecent'],
                                    case_res=settings['case_res'])
    bandits = uniform_kArmedBandits(**settings)
    anomalymap = bandits.get_preference_anomaly_map(CPT_Model)
    nE = np.shape(anomalymap)[2]
    imgs = []

    if nE == 1: Elabels = [' ($E(A)=0$)']
    else: Elabels = [f' ($E(A)=-\mu_E$)', ' ($E(A)=0$)', f' ($E(A)=\mu_E$)']
    for iE0 in range(nE):
        # attitude = bandits.get_attitude(anomalymap[:, :, iE0], hThresh=hSensitivity, lThresh=lSensitivity)
        # ax_labels = {'title': f'Preference Anomaly {Elabels[iE0]} \n{attitude}', 'x': '$r_{dist}$', 'y': '$p^{(+)}$'}
        ax_labels = {'title': f'Preference Anomaly {Elabels[iE0]}', 'x': '$r_{dist}$', 'y': '$p^{(+)}$'}
        imgs.append(bandits.plot_heatmap2D(axs[iE0], anomalymap[:, :, iE0], labels=ax_labels))
    clb = plt.colorbar(imgs[-1],ax=axs[-1])
    clb.ax.set_title('Certainty\n Preference')

    # fig.tight_layout()
    print(f'General Attitude: {bandits.get_attitude(anomalymap, hThresh=hSensitivity, lThresh=lSensitivity)}')


def redraw_anomaly(event,Widgets):
    print(f'Redrawing anomalies...',end='')
    global imgs
    global fig
    # global clb
    settings = {}
    # settings['p_split'] = Widgets.widgets['phat'].val
    settings['r_dist'] = Widgets.widgets['rhat'].val
    settings['Ecent'] = Widgets.widgets['ehat'].val
    settings['rationality'] = Widgets.widgets['lambda'].val
    settings['case_res'] = Widgets.widgets['resolution'].val

    CPT_params = {}
    CPT_params['b'] = Widgets.widgets['b'].val  # reference point
    CPT_params['gamma'] = Widgets.widgets['gamma'].val  # diminishing return gain
    CPT_params['lam'] = Widgets.widgets['ell'].val  # loss aversion
    CPT_params['alpha'] = Widgets.widgets['alpha'].val  # prelec parameter
    CPT_params['delta'] = Widgets.widgets['delta'].val  # Probability weight
    CPT_params['theta'] = Widgets.widgets['lambda'].val  # rationality

    hSensitivity = 0.25
    lSensitivity = 0.1

    bandits = uniform_kArmedBandits(**settings)
    # bandits = uniform_kArmedBandits()
    anomalymap = bandits.get_preference_anomaly_map(CPT_params)
    nE = np.shape(anomalymap)[2]
    print(f'[n={nE}]',end='')
    if nE == 1: Elabels = [' ($E(A)=0$)']
    else: Elabels = [' ($E(A)<0$)', ' ($E(A)=0$)', ' ($E(A)>0$)']
    for iE0 in range(nE):
        new_im = anomalymap[:, :, iE0]
        new_im = new_im + 0.001 if np.all(new_im)==0 else new_im
        # print(f'[{np.min(new_im)},{np.max(new_im)}]')
        if np.any(np.isnan(new_im)):  print(f'nanvals')
        if np.any(np.isinf(new_im)):  print(f'inf vals')
        imgs[iE0].set_data(new_im)
        # imgs[iE0].set_clim(vmin=-0.6, vmax=0.6)
        # imgs[iE0].autoscale()
        attitude = bandits.get_attitude(new_im, hThresh=hSensitivity, lThresh=lSensitivity)
        imgs[iE0].title  = f'Preference Anomaly {iE0} {Elabels[iE0]} \n{attitude}'
    fig.canvas.flush_events()
    fig.canvas.draw()
    # time.sleep(0.5)


    # clb = plt.colorbar(imgs[-1])
    print(f' [DONE]')


def reset(event,Widgets):
    for wkey in Widgets.widgets:
        try: Widgets.widgets[wkey].reset()
        except: pass


################################################################
################################################################
################################################################
################################################################
################################################################

# def max_likelyhood_est(Q,si_obs,ai_obs):
#     """
#     We can effectively ignore the prior and the evidence because — given the Wiki definition of a uniform prior
#     distribution — all coefficient values are equally likely. And probability of all data values (assume continuous)
#     are equally likely, and basically zero.
#     maxLL docs: https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f
#     scipy minimization docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#     """
#     # posterior = likelihood x prior / evidence
#
#     # Define Likelyhood Function ------------
#     def CPT_handler(params,Q,si_obs,ai_obs,const,param_names):
#         # UNPACK PARAMS
#         CPT_params = {}
#         for i,key in enumerate(param_names): CPT_params[key] = params[i] # add in design variables
#         for key in const: CPT_params[key] = const[key] # add in constants
#
#         # CALC PROBABILITY OF OBS ai UNDER MODEL
#         nObs = len(si_obs)
#         Pai_given_M = [humanDM(Q, si_obs[i], CPT_params)[ai_obs[i]] for i in range(nObs)]
#         return  -np.sum(np.log(Pai_given_M))
#         # return -np.prod(Pai_given_M)
#
#         # Pai_given_M = np.array([humanDM(Q, si_obs[i], CPT_params) for i in range(nObs)])
#         # return -np.sum(stats.norm.logpdf(ai_obs, loc=np.argmax(Pai_given_M,axis=1))) #, scale=sd
#
#     param_names = ['gamma','lam','alpha','delta']
#     guess_params = {}
#     guess_params['gamma'] = 0.5  # utility gain
#     guess_params['lam'] = 5  # relative weighting of gains/losses
#     guess_params['alpha'] = 0.5  # prelec parameter
#     guess_params['delta'] = 0.5  # convexity gain?
#
#     const_params = {}
#     const_params['b'] = None
#     const_params['theta'] = 10  # rationality
#
#     bounds_params = {}
#     bounds_params['b'] = None
#     bounds_params['gamma'] = BOUNDS01
#     bounds_params['lam'] = BOUNDS0inf
#     bounds_params['alpha'] = BOUNDS0inf
#     bounds_params['delta'] = BOUNDS01
#     bounds_params['theta'] = 10
#
#     # Define opt vars -------------------
#     SOLVER = 'Nelder-Mead'
#     guess = np.array([guess_params[key] for key in guess_params])
#     bounds = np.array([bounds_params[key] for key in guess_params])
#     obj_fun = partial(CPT_handler,Q=Q,si_obs=si_obs,ai_obs=ai_obs,const=const_params,param_names = param_names)
#     results = minimize(obj_fun, guess,bounds=bounds,method = SOLVER, options = {'disp': True})# method= 'Nelder - Mead'
#     params0 = results['x']
#
#     est_params = {}
#     for i,key in enumerate(param_names):  est_params[key] = params0[i] # design variables
#     for key in const_params:  est_params[key] = const_params[key]  # design variables
#     return est_params



def generate_test_cases():
    nS = 4
    nA = 2
    nO = 100
    r_penalty = -3
    p_penalty = 0.5
    range_cert = [0,18]
    range_uncert = [0,18]

    A = np.arange(nA)

    T = np.zeros([1,nA,nS])
    T[0, 0, :] = [1, 0, 0, 0]
    T[0, 1, :] = [0,0,1-p_penalty,p_penalty]
    Rn = np.zeros([nO, nS])
    En = np.zeros([nO, nA])
    for obs in range(nO):
        E_cert = randrange(range_cert[0],range_cert[1])
        E_uncert = randrange(range_uncert[0],range_uncert[1])
        # E_uncert = E_cert # verify no preference
        # r_loss = r_penalty
        # r_gain = (E_uncert - r_penalty*p_penalty)/(1-p_penalty)
        # Rn[obs] = [E_cert,0,r_gain,r_loss]
        Rn[obs] = [E_cert, 0, E_uncert-r_penalty, E_uncert+r_penalty]
        # En[obs] = [np.sum(Rn[obs]* T[0, ai, :]) for ai in A]
        # print(f'En = {En[obs]} \t Rn = {Rn[obs]} pd = {noisy_rational(En[obs],rationality=1)}')
    return Rn

def simulate(Rn,CPT_params):
    nS = 4
    nA = 2
    nO = np.shape(Rn)[0]
    p_penalty = 0.5
    A = np.arange(nA)
    T = np.zeros([1, nA, nS])
    T[0, 0, :] = [1, 0, 0, 0]
    T[0, 1, :] = [0, 0, 1 - p_penalty, p_penalty]

    CPT = CPT_Handler(**CPT_params)

    opt_pref = np.zeros([nO, nA])
    cpt_pref = np.zeros([nO, nA])
    cpt_choice = np.zeros(nO)
    for obs in range(nO):
        E_opt =np.array([np.sum(Rn[obs] * T[0, ai, :]) for ai in A])
        opt_pref[obs] = noisy_rational(E_opt, rationality=1)

        T_perc = np.zeros(np.shape(T))
        T_perc[0, 0, :] = CPT.prob_weight(T[0, 0, :])
        T_perc[0, 1, :] = CPT.prob_weight(T[0, 1, :])
        R_perc = CPT.utility_weight(Rn[obs])
        E_cpt =np.array( [np.sum(R_perc * T_perc[0, ai, :]) for ai in A])

        # print(E_cpt-E_opt)
        cpt_pref[obs] = noisy_rational(E_cpt, rationality=CPT.lam)

        cpt_choice[obs] = np.random.choice([0,1],p=cpt_pref[obs])
    return cpt_choice,cpt_pref,opt_pref

def MLE(Rn,a_obs,ASSUMPTION ):
    nS = 4
    nA = 2
    r_penalty = -3
    p_penalty = 0.5
    A = np.arange(nA)
    T = np.zeros([1, nA, nS])
    T[0, 0, :] = [1, 0, 0, 0]
    T[0, 1, :] = [0, 0, 1 - p_penalty, p_penalty]

    def OBJ(x,Rn,a_obs,idx,assumption):
        w_pen = 100
        nO = np.shape(Rn)[0]
        assumption_bnds = {}
        assumption_bnds['averse'] = [0.3,1]
        assumption_bnds['insensitive'] = [-0.3, 0.3]
        assumption_bnds['seeking'] = [-1, -0.3]

        # Load design parameters into dict
        CPT_params = {}
        for i,key in enumerate(idx.keys()):
            CPT_params[key] = x[i]
        CPT = CPT_Handler(**CPT_params)  # intiate CPT

        # Calculate probability of obs given model
        Pai_given_M = np.zeros([nO,1])
        opt_pref = np.zeros([nO, nA])
        cpt_pref = np.zeros([nO, nA])
        for obs in range(nO):
            E_opt = np.array([np.sum(Rn[obs] * T[0, ai, :]) for ai in A])
            opt_pref[obs] = noisy_rational(E_opt, rationality=1)

            T_perc = np.zeros(np.shape(T))
            T_perc[0, 0, :] = CPT.prob_weight(T[0, 0, :])
            T_perc[0, 1, :] = CPT.prob_weight(T[0, 1, :])
            R_perc = CPT.utility_weight(Rn[obs])
            E_cpt = np.array([np.sum(R_perc * T_perc[0, ai, :]) for ai in A])
            cpt_pref[obs] = noisy_rational(E_cpt, rationality=CPT.lam)
            Pai_given_M[obs] =  cpt_pref[obs,int(a_obs[obs])]

        # Calculate penalty =====================
        anomaly = (opt_pref- cpt_pref)[:,0]
        bnds = assumption_bnds[assumption]


        ilb = np.array(anomaly >= bnds[0],dtype='int8')
        iub =np.array(anomaly <= bnds[1],dtype='int8')
        inbnds = np.all(np.append(np.array([ilb]), np.array([iub]), axis=0),axis=0)
        dist_bnds = np.min([np.abs(anomaly - bnds[0]), np.abs(anomaly - bnds[1])], axis=0)
        dist_bnds[inbnds] = 0  # no penalty for assumed anomalies

        penalty = np.log(1+dist_bnds)
        penalty[inbnds] = 0
        penalty = w_pen * np.sum(penalty)
        penalty = 0
        nLL = -np.sum(np.log(Pai_given_M))
        return nLL + penalty


    # SOLVER = 'Nelder-Mead'

    idx = {}
    idx['b'] = 0
    idx['gamma'] = 1  # diminishing return gain
    idx['lam'] = 2 # loss aversion
    idx['alpha'] = 3  # prelec parameter
    idx['delta'] = 4  # Probability weight
    idx['theta'] = 5  # rationality

    n_params = len(idx.keys())
    guess = np.zeros(n_params)
    guess[idx['b']] = 0.0  # reference point
    guess[idx['gamma']] = 1.0  # diminishing return gain
    guess[idx['lam']] = 1.0  # loss aversion
    guess[idx['alpha']] = 1.0  # prelec parameter
    guess[idx['delta']] = 1.0  # Probability weight
    guess[idx['theta']] = 1.0  # rationality

    bounds = list(np.zeros(n_params))
    bounds[idx['b']]        = (-5.0,5.0)  # reference point
    bounds[idx['gamma']]    = (0,1.0)  # diminishing return gain
    bounds[idx['lam']]      = (1.0,9.0)  # loss aversion
    bounds[idx['alpha']]    = (0.0,10)  # prelec parameter
    bounds[idx['delta']]    = (0.0,1.0)  # Probability weight
    bounds[idx['theta']]    = (1.0,10.0)  # rationality


    const = (   {'type': 'eq', 'fun': lambda x: x[idx['b']] - 0  },    # = 0
                {'type': 'eq', 'fun': lambda x: x[idx['theta']] - 1 }, # = 0
                {'type': 'eq', 'fun': lambda x: x[idx['alpha']] - 1},  # = 0
                )


    obj_fun = partial(OBJ, Rn=Rn,a_obs =a_obs,idx =idx,assumption=ASSUMPTION)
    results = minimize(obj_fun, guess,
                       bounds=bounds,
                       constraints=const,
                       # method=SOLVER,
                       options={'disp': False})
    x0 = results['x']

    MLE_params = {}
    for i, key in enumerate(idx.keys()):
        MLE_params[key] = x0[i]
        # print(f'{key} = {x0[i]}')
    return MLE_params

def main():
    global fig
    # fig, ax = plt.subplots(2,2)
    # ASSUMPTION = 'insensitive'
    ASSUMPTION = 'averse'
    # ASSUMPTION = 'seeking'

    CPT_def = {}
    CPT_def['b'] = 0.0  # reference point
    CPT_def['gamma'] = 1.0  # diminishing return gain
    CPT_def['lam'] = 1.0  # loss aversion
    CPT_def['alpha'] = 1.0  # prelec parameter
    CPT_def['delta'] = 1.0  # Probability weight
    CPT_def['theta'] = 1.0  # rationality

    CPT_params = copy.copy(CPT_def)
    # CPT_params['b'] = 0.0  # reference point
    # CPT_params['gamma'] = 0.5  # diminishing return gain
    # CPT_params['lam'] = 8.0  # loss aversion
    # CPT_params['alpha'] = 3.0  # prelec parameter
    # CPT_params['delta'] = 1.0  # Probability weight
    # CPT_params['theta'] = 1.0  # rationality



    Rn = generate_test_cases()
    cpt_choice,cpt_pref,opt_pref = simulate(Rn, CPT_params)
    Est_params = MLE(Rn,a_obs=cpt_choice,ASSUMPTION=ASSUMPTION)

    data = {}
    data['TRUE'] = []
    data['MLE'] = []
    data['ERROR'] = []
    cols = CPT_params.keys()
    for i, key in enumerate(Est_params.keys()):
        data['TRUE'].append(CPT_params[key])
        data['MLE'].append(Est_params[key])
        data['ERROR'].append(CPT_params[key]-Est_params[key])

    df = pandas.DataFrame.from_dict(data,orient='index',columns=cols)
    print(df.round(2))














    # settings_def = {}
    # # settings['p_split'] = Widgets.widgets['phat'].val
    # settings_def['r_dist'] = 5.0
    # settings_def['Ecent'] = 10
    # settings_def['rationality'] = 1.0
    # settings_def['case_res'] = 50
    #
    # # CPT_def['b'] = 0.0  # reference point
    # # CPT_def['gamma'] = 0.802  # diminishing return gain
    # # CPT_def['lam'] = 3.0  # loss aversion
    # # CPT_def['alpha'] = 0.8  # prelec parameter
    # # CPT_def['delta'] = 0.8  # Probability weight
    # # CPT_def['theta'] = 1.0  # rationality
    #
    # Widgets     = PlotWidgets()
    # widget_caller = [
    #     # lambda axi: Widgets.add_text('CPT Reward Params',ax=axi),
    #     lambda axi: Widgets.add_text('CPT Parameters', ax=axi),
    #     lambda axi: Widgets.add_slider('Reward reference ($b$)',[-5, 5],ref='b',vinit=CPT_def['b'],ax=axi,step=1),
    #     lambda axi: Widgets.add_slider(r'Diminishing rewards ($\gamma$)', [0.01, 1],ref='gamma',vinit=CPT_def['gamma'],ax=axi,step=0.01),
    #     lambda axi: Widgets.add_slider(r'Loss aversion gain ($\ell$)', [1, 10],ref='ell',vinit=CPT_def['lam'],ax=axi,step=0.1),
    #
    #     # lambda axi: Widgets.add_text('CPT Probability Params ',ax=axi),
    #     lambda axi: Widgets.add_slider(r'Probability optimism ($\alpha$)', [0.1, 10],ref='alpha',vinit=CPT_def['alpha'],ax=axi,step=0.1),
    #     lambda axi: Widgets.add_slider(r'Probability sensitivity ($\delta$)', [0.01, 1],ref='delta',vinit=CPT_def['delta'],ax=axi,step=0.01),
    #     lambda axi: Widgets.add_slider(r'Policy rationality ($\lambda$)', [0.1, 10], ref='lambda', ax=axi,vinit=CPT_def['theta'], step=0.1),
    #
    #     lambda axi: Widgets.add_text('Environment Parameters', ax=axi),
    #     lambda axi: Widgets.add_slider('Plotting Resolution', [0, 100], ref='resolution', ax=axi,vinit=settings_def['case_res'],step=1),
    #     lambda axi: Widgets.add_slider('Expected reward ($\mu_{E}$)', [-30, 30], ref='ehat', ax=axi,vinit=settings_def['Ecent'],step=1),
    #     lambda axi: Widgets.add_slider(r'Deviation of payoffs ($\sigma_{r})$', [0.1, 10], ref='rhat', ax=axi,vinit=settings_def['r_dist'],step=0.1),
    #     # lambda axi: Widgets.add_slider(r'Prob. of loss ($p(o^{-}| a_{2}))$)', [0, 1], ref='phat', ax=axi),
    #
    #     lambda axi: Widgets.add_button('Reset to Optimal Params', ref='reset_button', ax=axi),
    #     lambda axi: Widgets.add_button('Redraw Anomalies',ref='redraw_button',ax=axi),
    # ]
    #
    # nWidgets = len(widget_caller)
    # widget_rows = len(widget_caller)
    # widget_cols = 1
    # grid_sz = (2 * nWidgets, 3)
    #
    # fig = plt.figure(figsize=(20, 12))
    # plt.subplots_adjust(hspace=0.333)
    #
    # w_rations = [0.3,0.3,0.4]
    # mainGrid = gridspec.GridSpec(2, 3,width_ratios=w_rations)
    #
    #
    # axR = plt.subplot(mainGrid[0, 0])
    # axP =  plt.subplot(mainGrid[0, 1])
    # widgetCell = mainGrid[0,2]
    # axEn =  plt.subplot(mainGrid[1, 0])
    # axE0 =  plt.subplot(mainGrid[1, 1])
    # axEp =  plt.subplot(mainGrid[1, 2])
    #
    #
    #
    # # Create widget array in grid cell
    #
    # widgetGrid = gridspec.GridSpecFromSubplotSpec(widget_rows, widget_cols, widgetCell)
    #
    #
    # widget_axes = []
    # for iw,wcall in enumerate(widget_caller):
    #     axW =plt.subplot(widgetGrid[iw, 0])
    #     # axW = plt.subplot2grid(grid_sz, (1, 2), rowspan=nWidgets)
    #     widget_axes.append(wcall(axW))
    #
    #
    # # Intialize plots ===============
    # plot_transforms(axR, axP, CPT_def)
    # # plot_anomaly([axEn,axE0,axEp], CPT_def)
    # plot_anomaly([axEn, axE0, axEp], Widgets)
    #
    #
    # button = widget_axes[-1]
    #
    # Widgets.widgets['b'].on_changed(partial(redraw, Widgets=Widgets))
    # Widgets.widgets['gamma'].on_changed(partial(redraw, Widgets=Widgets))
    # Widgets.widgets['ell'].on_changed(partial(redraw, Widgets=Widgets))
    # Widgets.widgets['alpha'].on_changed(partial(redraw, Widgets=Widgets))
    # Widgets.widgets['delta'].on_changed(partial(redraw, Widgets=Widgets))
    # Widgets.widgets['lambda'].on_changed(partial(redraw, Widgets=Widgets))
    #
    # Widgets.widgets['redraw_button'].on_clicked(partial(redraw_anomaly, Widgets=Widgets))
    # Widgets.widgets['reset_button'].on_clicked(partial(reset, Widgets=Widgets))
    #
    #
    # # plt.ax
    # plt.show()
if __name__ == "__main__":
    main()