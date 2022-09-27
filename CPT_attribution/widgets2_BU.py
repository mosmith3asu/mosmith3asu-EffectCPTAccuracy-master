import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons,TextBox
from agents.policies import noisy_rational,CPT_Handler
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
        print(self.farthest_xloc)
        label = widget.ax.get_children()[1]
        self.farthest_xloc = min(abs(self.farthest_xloc), np.min(label.xy[:, 0]))
        # plt.subplots_adjust(right= 1 - (self.widget_box_sz+self.farthest_xloc*2))
        # if self.loc == 'right':
        #     self.farthest_xloc = min(abs(self.farthest_xloc),np.min(label.xy[:, 0]))
        #     plt.subplots_adjust(left=self.widget_box_sz,right = 1 - self.farthest_xloc)
        # else:
        #     self.farthest_xloc = max(abs(self.farthest_xloc), np.max(label.xy[:,0]))
        #     plt.subplots_adjust(left= self.farthest_xloc)

    # def rescale_subplots(self):
    #     if self.loc == 'right':
    #         plt.subplots_adjust(right=1 - self.farthest_xloc)
    #     else:
    #         self.farthest_xloc = max(self.farthest_xloc, np.max(label.xy[:, 0]))
    #         plt.subplots_adjust(left=self.farthest_xloc)


    def add_slider(self,label,vrange,ref=None, vinit=None, height=0.03,step= 0.1):
        vinit = np.mean(vrange) if vinit is None else vinit
        ax = plt.axes([self.widget_box[0], self.y0, self.widget_box[2]-self.pad[1], height])
        slider = Slider(ax, label, vrange[0], vrange[1], valinit=vinit, valstep=step)
        ref = label if ref is None else ref
        self.widgets[ref] = slider
        self.adj_subplots(slider)
        self.y0 -= height
        return slider

    def add_button(self,label,ref=None,height=0.03):
        ax = plt.axes([self.widget_box[0], self.y0, self.widget_box[2]-self.pad[1], height])
        button = Button(ax, label) #color=axcolor, hovercolor='0.975'
        ref = label if ref is None else ref
        self.widgets[ref] = button
        self.y0 -= height
        return button

    def add_radio(self,label,options,ref=None,vinit=0,height=0.03):
        self.y0 -= height*(len(options)-0.5)
        ax = plt.axes([self.widget_box[0], self.y0, self.widget_box[2]-self.pad[1], height*len(options)])#, facecolor=axcolor
        radio = RadioButtons(ax,options, active=vinit)
        ref = label if ref is None else ref
        self.widgets[ref] = radio
        self.y0 -= height
        return radio

    def add_textbox(self,label,ref=None,vinit='',height=0.03):
        ax = plt.axes([self.widget_box[0], self.y0, self.widget_box[2]-self.pad[1], height])#, facecolor=axcolor
        text_box = TextBox(ax,label, initial=vinit)
        ref = label if ref is None else ref
        self.widgets[ref] = text_box
        self.adj_subplots(text_box)
        self.y0 -= height
        return text_box

    def add_text(self, text, ref=None, pady =0.02 ,
                 xadj = 0.0, height=0.03,bg = 'k',fg='w',align='left'):
        ax = plt.axes([self.widget_box[0]-xadj, self.y0-pady, self.widget_box[2] - self.pad[1]+xadj, height])  # , facecolor=axcolor
        text_box = TextBox(ax, '', initial=text,color = bg, hovercolor=bg,textalignment=align,label_pad=-0)
        text_box.set_active = False
        text_box.text_disp.set_color(fg)  # text inside the edit box
        text_box.text_disp.set_fontweight('bold')
        text_box.text_disp.set_verticalalignment('top')
        ref = text if ref is None else ref
        self.widgets[ref] = text_box
        self.y0 -=(height+pady)
        return text_box
    def vspace(self,height=0.03):
        self.y0 -= height

def plot_transforms(axR,axP,CPT_params,rrange = (-10,10),res=100):
    c_CPT = 'orange'
    c_OPT = 'black'
    c_GUIDE= 'lightgrey'

    CPT_policy = CPT_Handler(**CPT_params)

    r = np.linspace(rrange[0],rrange[1],res)
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
    # axR.set_aspect('equal', adjustable='box')
    # axR.set_aspect('equal')
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
    # axP.set_aspect('equal', adjustable='box')
    # axP.set_aspect('equal')

def main():
    fig, ax = plt.subplots(2,1)
    plt.subplots_adjust(left=0.05, right=0.5)

    # t = np.arange(0.0, 1.0, 0.001)
    # a0 = 5
    # f0 = 3
    # delta_f = 5.0
    # s = a0 * np.sin(2 * np.pi * f0 * t)
    # l, = plt.plot(t, s, lw=2)
    # ax.margins(x=0)

    CPT_def = {}
    CPT_def['b'] = 0  # reference point
    CPT_def['gamma'] = 1  # diminishing return gain
    CPT_def['lam'] = 1  # loss aversion
    CPT_def['alpha'] = 1  # prelec parameter
    CPT_def['delta'] = 1  # Probability weight
    CPT_def['theta'] = 1  # rationality
    axR = ax[0]
    axP = ax[1]
    plot_transforms(axR, axP, CPT_def)





    Widgets     = PlotWidgets()
    pheaderW    = Widgets.add_text('Environment Params')
    E0W         = Widgets.add_slider('Exp. reward center ($\mu_{E}$)', [-30, 30], ref='b')
    rhatW       = Widgets.add_slider(r'Deviation of payoffs ($\sigma_{r})$', [0.01, 10], ref='ell')
    phatW       = Widgets.add_slider(r'Prob. modification ($\hat{p}_{cert}$)', [-0.5, 0.5],step=0.05, ref='gamma')


    pheaderW    = Widgets.add_text('CPT Reward Params')
    bW          = Widgets.add_slider('Reference point ($b$)',[-30, 30],ref='b',vinit=CPT_def['b'])
    gammaW      = Widgets.add_slider(r'Diminishing sensitivity ($\gamma$)', [0.01, 1],ref='gamma',vinit=CPT_def['gamma'])
    ellW        = Widgets.add_slider(r'Loss aversion gain ($\ell$)', [1, 10],ref='ell',vinit=CPT_def['lam'])

    pheaderW    = Widgets.add_text('CPT Probability Params ')
    alphaW      = Widgets.add_slider(r'Probability optimism ($\alpha$)', [0.1, 10],ref='alpha',vinit=CPT_def['alpha'])
    deltaW      = Widgets.add_slider(r'Probability sensitivity ($\delta$)', [0.01, 1],ref='delta',vinit=CPT_def['delta'])

    pheaderW    = Widgets.add_text('Policy Params')
    lamW        = Widgets.add_slider(r'Rationality ($\lambda$)', [0.1, 30], ref='lambda')

    Widgets.vspace()
    resetbutton = Widgets.add_button('Redraw Plot')

    # radio = Widgets.add_radio('color',('red','blue','green'))



    # def update(val):
    #     amp = samp.val
    #     freq = sfreq.val
    #     l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    #     fig.canvas.draw_idle()
    #
    #
    # sfreq.on_changed(update)
    # samp.on_changed(update)

    # resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    # button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    # def reset(event):
    #     sfreq.reset()
    #     samp.reset()
    # button.on_clicked(reset)

    # rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    # radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
    #
    #
    # def colorfunc(label):
    #     l.set_color(label)
    #     fig.canvas.draw_idle()
    # radio.on_clicked(colorfunc)

    plt.show()
if __name__ == "__main__":
    main()