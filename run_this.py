from envs.env import ArmEnv
from algorithms.pd.PD import PD
from algorithms.pd.pd_controller import learn
import numpy as np

# set env
env = ArmEnv()

# parameters
algorithm_name = 'pd'
data_path = './Data/'
model_path = './model/' + algorithm_name + "/"

"""parameters for running"""
nb_epochs = 50
nb_epoch_cycles = 50
nb_rollout_steps = 300

file_name = '_epochs_' + str(nb_epochs)\
            + "_episodes_" + str(nb_epoch_cycles) + \
            "_rollout_steps_" + str(nb_rollout_steps)

data_path_reward = data_path + algorithm_name + "/" + file_name + 'reward'
data_path_steps = data_path + algorithm_name + "/" + file_name + 'steps'
data_path_states = data_path + algorithm_name + "/" + file_name + 'states'
data_path_times = data_path + algorithm_name + "/" + file_name + 'times'

model_name = file_name + 'model'

steps = []


def train():

    if algorithm_name == 'ddpg':
        from algorithms.ddpg.ddpg import learn
        learn(network='mlp',
              env=env,
              noise_type='normal_0.2',
              restore=False,
              nb_epochs=nb_epochs,
              nb_epoch_cycles=nb_epoch_cycles,
              nb_train_steps=60,
              nb_rollout_steps=nb_rollout_steps,
              data_path_reward=data_path_reward,
              data_path_steps=data_path_steps,
              data_path_states=data_path_states,
              data_path_times=data_path_times,
              model_path=model_path,
              model_name=model_name,
              )

    if algorithm_name == 'pd':
        from algorithms.pd.pd_controller import learn
        learn(
            controller=PD,
            env=env,
            nb_epochs=nb_epochs,
            nb_epoch_cycles=nb_epoch_cycles,
            nb_rollout_steps=nb_rollout_steps,
            data_path_reward=data_path_reward,
            data_path_steps=data_path_steps,
            data_path_states=data_path_states,
            data_path_times=data_path_times,
        )

    # if algorithm_name == 'ppo1'


if __name__ == '__main__':
    train()