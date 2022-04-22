from typing import List, Dict, Tuple
import random
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import animation

from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.mu_layer = nn.Linear(128, out_dim)
        self.log_std_layer = nn.Linear(128, out_dim)

        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> (torch.Tensor, Normal):
        x = F.relu(self.hidden1(state))

        mu = torch.tanh(self.mu_layer(x)) * 2
        log_std = F.softplus(self.log_std_layer(x))
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.out = nn.Linear(128, 1)

        initialize_uniformly(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        value = self.out(x)

        return value


class A2CAgent(object):
    def __init__(self, env: gym.Env, gamma: float, entropy_weight: float):
        self.env = env
        self.gamma = gamma
        self.entropy_weight = entropy_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.transition: list = list()

        self.total_step = 0
        self.writer = SummaryWriter('tensorboard')

        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action, dist = self.actor(state)

        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        state, log_prob, next_state, reward, done = self.transition

        mask = 1 - done
        next_state = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
        pred_value = self.critic(state)
        targ_value = reward + self.gamma * self.critic(next_state) * mask
        value_loss = F.smooth_l1_loss(pred_value, targ_value.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        advantage = (targ_value - pred_value).detach()
        policy_loss = -advantage * log_prob
        policy_loss += self.entropy_weight * -log_prob

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False

        actor_losses, critic_losses, scores = [], [], []
        state = self.env.reset()
        score = 0

        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            actor_loss, critic_loss = self.update_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            state = next_state
            score += reward

            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            if self.total_step % plotting_interval == 0:
                # self._plot(self.total_step, scores, actor_losses, critic_losses)
                self._log(self.total_step, scores, actor_losses, critic_losses)

        self._plot(self.total_step, scores, actor_losses, critic_losses)
        self.env.close()

    def test(self):
        self.is_test = True

        state = self.env.reset()
        done = False
        score = 0

        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        return frames

    def _log(self,
             frame_idx: int,
             scores: List[float],
             actor_losses: List[float],
             critic_losses: List[float],
             ):
        """logging the training progresses."""
        actor_loss = np.mean(actor_losses[-10:])
        critic_loss = np.mean(critic_losses[-10:])
        score = np.mean(scores[-10:])
        self.writer.add_scalar('actor_loss', actor_loss, global_step=frame_idx)
        self.writer.add_scalar('critic_loss', critic_loss, global_step=frame_idx)
        self.writer.add_scalar('score', score, global_step=frame_idx)
        print(
            "frame: {} | Score: {} | ActorLoss: {} | CriticLoss: {}".format(frame_idx, score, actor_loss, critic_loss))

    def _plot(self, frame_idx: int, scores: List[float], actor_losses: List[float], critic_losses: List[float]):
        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        figure = plt.figure(figsize=(18, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)

        plt.show()


if __name__ == '__main__':
    def frame_gif(frames: List[np.ndarray]) -> None:
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames=len(frames), interval=50
        )
        plt.show()


    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_frames = 100000
    gamma = 0.9
    entropy_weight = 1e-2

    env_id = 'Pendulum-v0'
    env = gym.make(env_id)

    agent = A2CAgent(env, gamma, entropy_weight)

    agent.train(num_frames)

    frames = agent.test()

    frame_gif(frames)
