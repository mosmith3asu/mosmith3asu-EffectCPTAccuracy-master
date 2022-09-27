import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def main():


class Plotter(object):
    def __init__(self,plt,widget_box_loc='right',widgets_box_sz=0.5):
        self.grid_sz =  (2,2)
        # self.widgets = np.empty(self.grid_sz,dtype=object)

        self.widget_template = {}
        self.widget_template['label'] = str
        self.widget_template['obj'] = object
        self.widget_template['update'] = bool


        plt.subplots_adjust(**{widget_box_loc: widgets_box_sz})


        self.widgets = []

    def add_slider(self,label,range,update=False,orientation = 'horizontal'):
        slider = Slider(
            ax=axamp, label=label,
            valmin=range[0], valmax=range[1],
            valinit=init_amplitude,
            orientation= orientation
        )
        widget = copy.copy(self.widget_template)
        widget['label'] = label
        widget['obj'] = slider
        widget['update'] = update
        slider.on_changed(self.update())
        self.widgets.append(widget)

    def update(self):
        line.set_ydata(f(t, amp_slider.val, freq_slider.val))
        fig.canvas.draw_idle()

if __name__ == "__main__":
    main()



# The parametrized function to be plotted
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

t = np.linspace(0, 1, 1000)

# Define initial parameters
init_amplitude = 5
init_frequency = 3

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)
ax.set_xlabel('Time [s]')


# plt_adjust = {}
# plt_adjust['left'] = 0.25
# plt_adjust['right'] = 0.25
# plt_adjust['top'] = 0.25
# plt_adjust['bottom'] = 0.25


# adjust the main plot to make room for the sliders
# plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.1,
    valmax=30,
    valinit=init_frequency,
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()
