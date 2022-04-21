import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vizdoom import *
from torch.utils.tensorboard import SummaryWriter

import random
from datetime import datetime
import os
from skimage import transform

from collections import deque

import warnings

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding='valid')
        nn.init.xavier_normal_(self.conv2.weight)
        self.bn3 = nn.BatchNorm2d(128)

        def conv2d_size_out(size, kernel_size=4, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=8, stride=4)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=8, stride=4)))
        linear_input_size = convw * convh * 128

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

        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.action_mapping = {
            '[0]': [1, 0, 0],
            '[1]': [0, 1, 0],
            '[2]': [0, 0, 1]
        }

    def choose_action(self, explore_start, explore_stop, decay_rate, decay_step, state, actions):
        exp_exp_tradeoff = np.random.rand()

        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        if explore_probability > exp_exp_tradeoff:
            action = random.choice(actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2)
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = self.action_mapping[str(action)]

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


def create_environment():
    game = DoomGame()
    game.load_config("../../course/DQN-with-doom/basic.cfg")
    game.set_doom_scenario_path("../../course/DQN-with-doom/basic.wad")
    game.init()

    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions


def preprocess_frame(frame):
    # greyscale frame already done in vizdoom config
    cropped_frame = frame[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255.0
    preprocess_frame = transform.resize(normalized_frame, [84, 84])

    return preprocess_frame


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

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
    # exploration
    game.new_episode()
    print("Collecting experience...")
    for i in range(pretrain_length):
        if i == 0:
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        action = random.choice(possible_actions)
        reward = game.make_action(action)
        done = game.is_episode_finished()

        if done:
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            memory.add((state, action, reward, next_state, done))
            state = next_state
    tensorboard_path = '../../course/DQN-with-doom/tensorboard/'
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
    writer = SummaryWriter('../../course/DQN-with-doom/tensorboard/{}'.format(datetime.now()))
    print("training...")

    decay_step = 0
    game.init()

    for episode in range(start_episode + 1, total_episodes):
        step = 0
        episode_rewards = []

        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while step < max_steps:
            step += 1
            decay_step += 1

            action, explore_probability = dqn.choose_action(explore_start, explore_stop, decay_rate, decay_step,
                                                            state,
                                                            possible_actions)
            reward = game.make_action(action)
            done = game.is_episode_finished()
            episode_rewards.append(reward)
            loss = dqn.learn()

            if done:
                next_state = np.zeros((84, 84), dtype=np.int)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                step = max_steps
                memory.add((state, action, reward, next_state, done))
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                memory.add((state, action, reward, next_state, done))
                state = next_state

        total_reward = np.sum(episode_rewards)

        print('Episode: {}'.format(episode),
              'Total reward: {}'.format(total_reward),
              'Training loss: {:.4f}'.format(loss),
              'Explore P: {:.4f}'.format(explore_probability))

        writer.add_scalar('reward', total_reward, global_step=episode)
        writer.add_scalar('train-loss', loss, global_step=episode)

        if (episode + 1) % checkpoint_interval == 0:
            checkpoint = {"eval_net_state_dict": dqn.eval_net.state_dict(),
                          "target_net_state_dict": dqn.target_net.state_dict(),
                          "optimizer_state_dict": dqn.optimizer.state_dict(),
                          "episode": episode}
            path_checkpoint = "../../course/DQN-with-doom/checkpoint/"
            if not os.path.exists(path_checkpoint):
                os.mkdir(path_checkpoint)
            path_checkpoint = path_checkpoint + "checkpoint_episode_{}.pkl".format(episode)
            torch.save(checkpoint, path_checkpoint)


if __name__ == '__main__':
    game, possible_actions = create_environment()
    n_actions = game.get_available_buttons_size()
    h = 84
    w = 84
    learning_rate = 0.0002

    stack_size = 4
    stacked_frames = deque([np.zeros((h, w), dtype=np.int) for i in range(stack_size)], maxlen=4)

    total_episodes = 500
    max_steps = 100
    batch_size = 64
    target_update = 200

    explore_start = 1.0
    explore_stop = 0.01
    decay_rate = 0.0001

    gamma = 0.95
    pretrain_length = batch_size
    memory_size = 1000000
    training = True
    checkpoint_interval = 10
    checkpoint = False

    dqn = DQN(h, w, n_actions, learning_rate)
    memory = Memory(memory_size)

    if checkpoint:
        episode = 0
        path_checkpoint = "../../course/DQN-with-doom/checkpoint/checkpoint_episode_{}.pkl".format(episode)
        checkpoint = torch.load(path_checkpoint)

        dqn.eval_net.load_state_dict(checkpoint['eval_net_state_dict'])
        dqn.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        train(stacked_frames, start_episode=start_episode)
    else:
        train(stacked_frames)
