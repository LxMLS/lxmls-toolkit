import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
import math
import matplotlib.pyplot as plt


class PolicyGradient(nn.Module):

    def __init__(self):
        super(PolicyGradient, self).__init__()
        self.linear = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 2)

    def forward(self, state):

        # ----------
        # Solution to Exercise 6.4

        # raise Exception("Exercise 6.4")

        input1 = torch.autograd.Variable(torch.FloatTensor([state]))
        return F.log_softmax(self.linear2(F.sigmoid(self.linear(input1))))

        # End of solution to Exercise 6.4
        # ----------


def train():

    env = gym.make('CartPole-v0')

    env.seed(1)

    print("env.action_space", env.action_space)
    print("env.observation_space", env.observation_space)
    print("env.observation_space.high", env.observation_space.high)
    print("env.observation_space.low", env.observation_space.low)

    RENDER_ENV = False
    EPISODES = 5000
    rewards = []
    RENDER_REWARD_MIN = 50

    criterion = torch.nn.NLLLoss()
    policy = PolicyGradient()
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=0.002)
    env.reset()
    decay = 1
    resultlist = []
    for episode in range(EPISODES):
        observations = []
        observation = env.reset()
        while True:

            action = int(np.random.choice(
                range(2),
                p=np.exp(policy(observation).data.numpy()[0]))
            )
            observation_, reward, finished, info = env.step(action)
            observations.append((observation, action, reward, observation_))
            observation = observation_

            if finished:
                rewardlist = [x[2] for x in observations]
                cumulative = 0
                savelist = []
                for rew in rewardlist[::-1]:
                    cumulative = cumulative*decay + rew/200
                    savelist.append(cumulative)
                savelist = savelist[::-1]

                resultlist.append(savelist[0])
                if episode % 50 == 0:
                    plt.plot(resultlist)
                    plt.show()
                savelist = np.array(savelist)

                for (observation, action, reward, next_observation), cum_reward in zip(observations, savelist):
                    action = torch.autograd.Variable(
                        torch.LongTensor([action])
                    )
                    result = policy(observation)
                    loss = criterion(result, action)
                    (loss * cum_reward).backward()
                    optimizer.step()
                    optimizer.zero_grad()
                break
