# Setting according to the defintion of problem
import numpy as np
import copy as cp
from algorithms.pd.PD import PD
pd = PD()




def reward_step(state, safe_or_not, step_num):
    """ get reward at each step"""
    done = False

    # the target depth in z depth
    set_insert_goal_depth = 47
    peg_length = 50
    step_max = 200
    force = state[6:12]

    if safe_or_not is False:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Max_force_moment:', force)
        reward = -1 + (peg_length - state[2])/set_insert_goal_depth
        print("-------------------------------- The force is too large!!! -----------------------------")
    else:
        """consider force and moment"""
        reward = -0.02

    # insert complete
    if (peg_length + state[2]) > set_insert_goal_depth:
        print("+++++++++++++++++++++++++++++ The Assembly Phase Finished!!! ++++++++++++++++++++++++++++")
        reward = 1 - step_num / step_max
        done = True

    return reward, done


# this function adjust the output of the network in to usable actions
def actions(s, a, en_pd):
    if en_pd:
        action = pd.cal(s, np.array([0, 0, 15, 0, 0, 0]))
        action = action + action * a[0]
    else:
        action = a[0]
    action[0] /= 5000
    action[1] /= 5000
    action[2] /= 3000

    action[3] /= 1000
    action[4] /= 1000
    action[5] /= 1000

    if action[2] < 0:
        action[2] = 0

    for i in range(6):
        action[i] = round(action[i], 4)
    return action


# this function checks if the force and torque extends safety value
def safetycheck(s):
    if s[6] >= 200 or s[7]>= 200 or s[8] >= 200:
        return False
    elif s[9] >= 20 or s[10]>= 20 or s[11] >= 20:
        return False
    else:
        return True


def code_state(current_state):
    state = cp.deepcopy(current_state)

    """normalize the state"""
    position_scale = 1000
    final_state = state
    final_state[0:3] *= position_scale # m to mm
    for i in range(12):
        final_state[i] = round(final_state[i], 4)


    '''Add Threshold'''

    return final_state


def clear():
    pd.clear()

