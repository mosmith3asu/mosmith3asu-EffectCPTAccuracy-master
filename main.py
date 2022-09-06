# This is a sample Python script.
import math
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def rDist(statei):
    order = 2
    scale_closer = 2
    d0 = math.dist(statei[0, 0:2], statei[2, 0:2])
    d1 = math.dist(statei[1, 0:2], statei[2, 0:2])
    r = pow(scale_closer * min(d0, d1),order) + pow(max(d0, d1),order)
    return r

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    s1 = [1,0,1]
    s2 = [0,3,1]
    evader =  np.array([3,0,0])
    si_dmax = np.array([[1, 1, 0], [1, 1, 0], [5, 5, 0]])
    rmax = rDist(si_dmax)

    right = np.array([1,0,0])
    left = np.array([-1, 0, 0])
    up = np.array([0, -1, 0])
    down = np.array([0, 1, 0])


    state0 = np.array([s1,s2,evader])
    state1 = np.array([s1,s2,evader+right])
    stateW1 = np.array([s1,s2,evader+up])
    stateW2 = np.array([s1,s2,evader+up])
    print(f'state0 = {rDist(state0)/rmax}')
    print(f'state(best) = {rDist(state1)/rmax} >> {[rDist(stateW1)/rmax,rDist(stateW2)/rmax]}')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
