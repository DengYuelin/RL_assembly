# Setting according to the defintion of problem
import numpy as np
import copy as cp
from algorithms.pd.PD import PD
pd = PD()

def reword(s, timer):
    r = -(abs(s[0]) + abs(s[1]) + abs(s[2])) * 10
    if abs(s[0]) < 0.001 and abs(s[1]) < 0.001 and abs(s[2]) < 0.001:
        done = True
        r += (200 - timer)
    else:
        done = False
    return r, done


# this function adjust the output of the network in to usable actions
def actions(s, a, mode, en_pd):
    if en_pd:
        action = pd.cal(s, np.array([0, 0, -4, 0, 0]))
        action = action + action * a[0]
    else:
        action = a[0]
    if mode:
        a_a = action * 0.001
    else:
        a_a = action * 0.01
    return a_a


# this function checks if the force and torque extends safety value
def safetycheck(s):
    if s[3] >= 10 or s[4]>= 10 or s[5] >= 10:
        return False
    elif s[6] >= 10 or s[7]>= 10 or s[8] >= 10:
        return False
    else:
        return True

def code_state(current_state):
    state = cp.deepcopy(current_state)

    """normalize the state"""
    position_scale = 0.1
    torque_scale = 0.01
    final_state = state
    final_state[0:3] /= position_scale
    final_state[6:9] /= torque_scale

    '''Add Threshold'''

    return final_state

