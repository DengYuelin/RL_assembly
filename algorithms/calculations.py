# Setting according to the defintion of problem
import numpy as np
import copy as cp

def reword(s):
    r = -(abs(s[0]) + abs(s[1]) + abs(s[2]))*10
    if abs(s[0]) < 0.001 and abs(s[1]) < 0.001 and abs(s[2]) < 0.001:
        r = 1
    if abs(s[0]) < 0.001 and abs(s[1]) < 0.001 and abs(s[2]) < 0.001:
        done = True
    else:
        done = False
    return r, done


# this function adjust the output of the network in to usable actions
def actions(a, mode):  # here a ∈ action_bound
    if mode:
        a_a = a[0] * 0.001
    else:
        a_a = a[0] * 0.01
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
    scale = 0.1
    final_state = state / scale

    '''Add Threshold'''

    return final_state

