import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

# 重复性
np.random.seed(1)
torch.manual_seed(1)


class Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class PolicyGradient(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Network(self.n_features, self.n_actions).to(self.device)
        self.state_v = Network(self.n_features, 1).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def choose_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        prob_weight = self.model(observation)
        action = Categorical(logits=prob_weight).sample().item()

        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # 折扣和规范化每一个ep中的奖励
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        state = torch.as_tensor(np.vstack(self.ep_obs), dtype=torch.float32)
        action = torch.as_tensor(self.ep_as, dtype=torch.long)
        q = torch.as_tensor(discounted_ep_rs_norm)
        v = self.state_v(state)

        advantage = q - v

        action_logits = self.model(state)
        logp = F.cross_entropy(action_logits, action)
        # logp = Categorical(logits=action_logits).log_prob(action)
        loss = -(logp * advantage).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self, norm=True):
        # 对ep中的奖励进行折扣
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 反序遍历
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 规范化回报值
        if norm:
            discounted_ep_rs -= np.mean(discounted_ep_rs)
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
