import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

import numpy as np
import gym

np.random.seed(2)
torch.manual_seed(2)  # 指定随机种子,方便结果复现

# 超参数
OUTPUT_GRAPH = False  # 输出Graph
MAX_EPISODE = 3000  # 最大片段
DISPLAY_REWARD_THRESHOLD = 200  # 如果时间段的回报大于此阈值,渲染环境
MAX_EP_STEPS = 1000  # 在一个时间段内最大的时间步
RENDER = False  # 渲染耗费时间
GAMMA = 0.9  # 折扣因子
LR_A = 0.001  # 演员网络的学习率
LR_C = 0.01  # 评论家网络的学习率

env = gym.make('CartPole-v0')  # 加载环境CartPole
env.seed(1)  # 可复现的种子值
env = env.unwrapped  # 取消限制

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Actor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_features, 20),
            nn.ReLU(),
            nn.Linear(20, n_actions),
            nn.Softmax(dim=-1)
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0, 0.1)
                nn.init.constant_(layer.bias.data, 0.1)

    def forward(self, x):
        return Categorical(probs=self.layers(x))


class Critic(nn.Module):
    def __init__(self, n_features):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0, 0.1)
                nn.init.constant_(layer.bias.data, 0.1)

    def forward(self, x):
        return self.layers(x)


class AC(object):
    def __init__(
            self,
            n_features,
            n_actions,
            lr_a=0.001,
            lr_c=0.005,
            gamma=0.9
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Actor = Actor(self.n_features, self.n_actions).to(self.device)
        self.Critic = Critic(self.n_features).to(self.device)
        self.optimizer_a = optim.Adam(self.Actor.parameters(), lr=self.lr_a)
        self.optimizer_c = optim.Adam(self.Critic.parameters(), lr=self.lr_c)

    def choose_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        prob = self.Actor(observation)
        action = prob.sample().item()
        # print(observation, prob, action)

        return action

    def critic_learn(self, s, r, s_):
        s = torch.as_tensor(s, dtype=torch.float32)
        s_ = torch.as_tensor(s_, dtype=torch.float32)
        pred = self.Critic(s)
        v_ = self.Critic(s_)
        target = r + self.gamma * v_
        loss = F.smooth_l1_loss(pred, target.detach())

        self.optimizer_c.zero_grad()
        loss.backward()
        self.optimizer_c.step()

        return (target - pred).detach()

    def actor_learn(self, s, a, td_error):
        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.int32)
        logp = self.Actor(s).log_prob(a)
        loss = -(logp * td_error).mean()

        self.optimizer_a.zero_grad()
        loss.backward()
        self.optimizer_a.step()


RL = AC(
    N_F,
    N_A,
    LR_A,
    LR_C,
    GAMMA
)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = RL.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = RL.critic_learn(s, r, s_)
        RL.actor_learn(s, a, td_error)

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
