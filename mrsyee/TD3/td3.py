import copy
import os
import random
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import animation
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool
    ):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.max_size, self.size + 1)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )

    def __len__(self) -> int:
        return self.size


class GaussianNoise:
    """Gaussian Noise."""

    def __init__(
            self,
            action_dim: int,
            min_sigma: float = 1.0,
            max_sigma: float = 1.0,
            decay_period: int = 1000000
    ):
        """Initialize."""
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        """Get an action with gaussian noise."""
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )

        return np.random.normal(0, sigma, size=self.action_dim)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_w: float = 3e-3):
        """Initialize."""
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()

        return action


class Critic(nn.Module):
    def __init__(self, in_dim: int, init_w: float = 3e-3):
        """Initialize."""
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


class TD3Agent:
    """TD3Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        actor1 (nn.Module): target actor model to select actions
        actor2 (nn.Module): target actor model to select actions
        actor_target1 (nn.Module): actor model to predict next actions
        actor_target2 (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic1 (nn.Module): critic model to predict state values
        critic2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        exploration_noise (GaussianNoise): gaussian noise for policy
        target_policy_noise (GaussianNoise): gaussian noise for target policy
        target_policy_noise_clip (float): clip target gaussian noise
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        policy_update_freq (int): update actor every time critic updates this times
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            gamma: float = 0.99,
            tau: float = 5e-3,
            exploration_noise: float = 0.1,
            target_policy_noise: float = 0.2,
            target_policy_noise_clip: float = 0.5,
            initial_random_steps: int = int(1e4),
            policy_update_freq: int = 2
    ):
        """Initialize."""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.writer = SummaryWriter('tensorboard')

        # noise
        self.exploration_noise = GaussianNoise(
            action_dim, exploration_noise, exploration_noise
        )
        self.target_policy_noise = GaussianNoise(
            action_dim, target_policy_noise, target_policy_noise
        )
        self.target_policy_noise_clip = target_policy_noise_clip

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target1 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target2 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # concat critic parameters to use one optim
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(critic_parameters, lr=1e-3)

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # update step for actor
        self.update_step = 0

        # mode
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = (
                self.actor(torch.as_tensor(state, dtype=torch.float32).to(self.device))[0]
                    .detach().cpu().numpy()
            )

        # add noise for exploration during training
        if not self.is_test:
            noise = self.exploration_noise.sample()
            selected_action = np.clip(
                selected_action + noise, -1.0, 1.0
            )

            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> Tuple:
        """Update the model by gradient descent."""
        device = self.device

        sample = self.memory.sample_batch()
        states = torch.as_tensor(sample["obs"], dtype=torch.float32).to(device)
        next_states = torch.as_tensor(sample["next_obs"], dtype=torch.float32).to(device)
        actions = torch.as_tensor(sample["acts"].reshape(-1, 1), dtype=torch.float32).to(device)
        rewards = torch.as_tensor(sample["rews"].reshape(-1, 1), dtype=torch.float32).to(device)
        dones = torch.as_tensor(sample["done"].reshape(-1, 1), dtype=torch.float32).to(device)
        masks = 1 - dones

        # get actions with noise
        noise = torch.as_tensor(self.target_policy_noise.sample(), dtype=torch.float32).to(device)
        clipped_noise = torch.clamp(noise, -self.target_policy_noise_clip, self.target_policy_noise_clip)

        next_actions = (self.actor_target(next_states) + clipped_noise).clamp(-1.0, 1.0)

        next_values1 = self.critic_target1(next_states, next_actions)
        next_values2 = self.critic_target2(next_states, next_actions)
        next_values = torch.min(next_values1, next_values2)

        curr_returns = rewards + self.gamma * next_values * masks
        curr_returns = curr_returns.detach()

        values1 = self.critic1(states, actions)
        values2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(values1, curr_returns)
        critic2_loss = F.mse_loss(values2, curr_returns)

        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_step % self.policy_update_freq == 0:
            # train actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target_update
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.item(), critic_loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        actor_losses = []
        critic_losses = []
        scores = []
        score = 0

        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if (
                    len(self.memory) >= self.batch_size
                    and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            # plotting
            if self.total_step % plotting_interval == 0:
                self._log(self.total_step, scores, actor_losses, critic_losses)

        self._plot(self.total_step, scores, actor_losses, critic_losses)
        self.env.close()

    def test(self):
        """Test the agent."""
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

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target1.parameters(), self.critic1.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target2.parameters(), self.critic2.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

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

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            actor_losses: List[float],
            critic_losses: List[float],
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(30, 5))
        plt.subplot(131)
        plt.title("frame %s. score: %s" % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title("actor_loss")
        plt.plot(actor_losses)
        plt.subplot(133)
        plt.title("critic_loss")
        plt.plot(critic_losses)
        plt.show()


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1,1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


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

    env_id = "Pendulum-v0"
    env = gym.make(env_id)
    env = ActionNormalizer(env)
    env.seed(seed)

    num_frames = 50000
    memory_size = 100000
    batch_size = 128
    initial_random_steps = 10000

    agent = TD3Agent(
        env,
        memory_size,
        batch_size,
        initial_random_steps=initial_random_steps
    )

    agent.train(num_frames)

    frames = agent.test()

    frame_gif(frames)

