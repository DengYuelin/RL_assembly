from envs.vrepenv import ArmEnv
from algorithms.PD import PD
from algorithms.ddpg.ddpg import learn
import numpy as np
import time

MAX_EPISODES = 900
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
# set RL method
pd = PD()

file_name = 'test_run'
data_path = '../Data/prediction_data/'
model_path = '../Data/prediction_model/'

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
          nb_epoch_cycles=2,
          nb_train_steps=5,
          nb_rollout_steps=5)


if ON_TRAIN:
    train()
else:
    runpd()
