import retro
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from collections import deque

import random
import warnings

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = retro.make(game='SpaceInvaders-Atari2600')


# observation_space (210, 160, 3)
# action_space 8


class Memory(object):
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]


class Net(nn.Module):
    def __init__(self, h, w, outputs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding='valid')
        nn.init.xavier_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding='valid')
        nn.init.xavier_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding='valid')
        nn.init.xavier_normal_(self.conv2.weight)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=4, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=8, stride=4)), kernel_size=3, stride=2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=8, stride=4)), kernel_size=3, stride=2)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(512, outputs)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = x.to(device)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class DQN(object):
    def __init__(self, h, w, n_actions, lr):
        self.eval_net = Net(h, w, n_actions)
        self.target_net = Net(h, w, n_actions)
        self.learn_step_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def choose_action(self, explore_start, explore_stop, decay_rate, decay_step, state, actions):
        exp_exp_tradeoff = np.random.rand()

        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        if explore_probability > exp_exp_tradeoff:
            action = random.choice(actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2)
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()

        return action, explore_probability

    def learn(self):
        if self.learn_step_counter % target_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch = memory.sample(batch_size)
        states_mb = torch.FloatTensor(np.array([each[0] for each in batch], ndmin=3)).permute(0, 3, 1, 2)
        actions_mb = torch.LongTensor(np.array([each[1] for each in batch]))
        rewards_mb = torch.FloatTensor(np.array([each[2] for each in batch]))
        rewards_mb = rewards_mb.unsqueeze(1)
        next_states_mb = torch.FloatTensor(np.array([each[3] for each in batch], ndmin=3)).permute(0, 3, 1, 2)

        q_eval = self.eval_net(states_mb).gather(1, actions_mb)
        q_next = self.target_net(next_states_mb).detach()
        q_target = rewards_mb + gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))


def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    return preprocessed_frame  # 110*84*1


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


def train(stacked_frames, start_episode=-1):
    print("Collecting experience...")
    for i in range(pretrain_length):
        if i == 0:
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]
        next_state, reward, done = env.step(action)

        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        if done:
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            memory.add((state, action, reward, next_state, done))
            state = next_state


if __name__ == '__main__':
    h = 110
    w = 84
    action_size = env.action_space.n
    possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
    learning_rate = 0.00025

    total_episodes = 50
    max_steps = 50000
    batch_size = 64

    explore_start = 1.0
    explore_stop = 0.01
    decay_rate = 0.00001

    gamma = 0.9

    pretrain_length = batch_size
    memory_size = 1000000

    stack_size = 4
    stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

    training = False
    episode_render = False
    checkpoint_interval = 10
    checkpoint = False

    dqn = DQN(h, w, action_size, learning_rate)
    memory = Memory(memory_size)

    if checkpoint:
        episode = 0
        path_checkpoint = "../../course/DQN-with-atari/checkpoint/checkpoint_episode_{}.pkl".format(episode)
        checkpoint = torch.load(path_checkpoint)

        dqn.eval_net.load_state_dict(checkpoint['eval_net_state_dict'])
        dqn.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        train(stacked_frames, start_episode=start_episode)
    else:
        train(stacked_frames)
