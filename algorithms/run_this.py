from envs.vrepenv import ArmEnv
from algorithms.pd.PD import PD
from algorithms.ddpg.ddpg import learn
import numpy as np

MAX_EPISODES = 900
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()

# ================================================================
algorithm_name = 'ddpg'
data_path = '../Data/'
model_path = '../model/'

"""parameters for running"""
nb_epochs = 5
nb_epoch_cycles = 100
nb_rollout_steps = 200

file_name = '_epochs_' + str(nb_epochs)\
            + "_episodes_" + str(nb_epoch_cycles) + \
            "_rollout_steps_" + str(nb_rollout_steps)

data_path_reward = data_path + algorithm_name + file_name + 'reward'
data_path_steps = data_path + algorithm_name + file_name + 'steps'
data_path_states = data_path + algorithm_name + file_name + 'states'
data_path_times = data_path + algorithm_name + file_name + 'times'

data_path_model = model_path + algorithm_name + file_name + 'model'

steps = []


def runpd():
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):

            a = pd.cal(s, np.array([0, 0, -4, 0, 0]))
            s, r, done, safe = env.step(a)
            if done or j == MAX_EP_STEPS - 1 or safe is False:
                print('Ep: %i | %s | %s | step: %i' % (
                    i, '---' if not done else 'done', 'unsafe' if not safe else 'safe', j))
                break


def train():
    # start training
    learn(network='mlp',
          env=env,
          data_path=data_path,
          noise_type='normal_0.2',
          file_name=file_name,
          model_path=model_path,
          restore=False,
          nb_epochs=1,
          nb_epoch_cycles=10,
          nb_train_steps=60,
          nb_rollout_steps=200)

    if algorithm_name == 'ddpg':
        learn(network='mlp',
              env=env,
              noise_type='normal_0.2',
              restore=False,
              nb_epochs=1,
              nb_epoch_cycles=10,
              nb_train_steps=60,
              nb_rollout_steps=200,
              data_path_reward=data_path_reward,
              data_path_steps=data_path_steps,
              data_path_states=data_path_states,
              data_path_times=data_path_times,
              model_path=data_path_model,
              )


if __name__ == '__main__':

    if ON_TRAIN:
        train()
    else:
        runpd()
