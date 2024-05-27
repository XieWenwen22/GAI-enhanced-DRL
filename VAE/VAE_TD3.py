import argparse
from collections import namedtuple
from copy import deepcopy
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from VAE import VAE
from Environment_relay_simple_state import Environment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="Pendulum-v0")  # OpenAI gym environment name， BipedalWalker-v2
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=50000, type=int) # replay buffer size
parser.add_argument('--num_iteration', default=100000, type=int) #  num of  games
parser.add_argument('--batch_size', default=128, type=int) # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=2000, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = Environment()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
hidden_dim = 128
latent_dim = 32
min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name + args.env_name +'./'
'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

        self.max_action = 1

    def forward(self, state):
        a = F.tanh(self.fc1(state))
        a = F.tanh(self.fc2(a))
        # print(a)
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3():
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim,max_action):

        self.actor = Actor(latent_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(latent_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(latent_dim, action_dim).to(device)
        self.critic_1_target = Critic(latent_dim, action_dim).to(device)
        self.critic_2 = Critic(latent_dim, action_dim).to(device)
        self.critic_2_target = Critic(latent_dim, action_dim).to(device)
        self.vae = VAE(state_dim,hidden_dim,latent_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),lr=args.learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),lr=args.learning_rate)
        self.vae_optimizer = optim.Adam(self.vae.parameters(),lr=1e-3)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        mu, log_var = self.vae.encode(state)
        encode_state = self.vae.reparameterization(mu, log_var)
        # print("Encode state:",encode_state.shape)
        return self.actor(encode_state).cpu().data.numpy().flatten()

    def _invert_gradients(self, grad, vals, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)

        max_p = torch.from_numpy(np.array([1]))
        min_p = torch.from_numpy(np.array([-1]))
        rnge = max_p-min_p


        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def update(self, num_iteration):

        for i in range(1):
            x, y, u, r, d = self.memory.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            state_hat, mu, log_var = self.vae(state)
            vae_loss, _, _ = self.vae.loss_function(state,state_hat,mu,log_var)
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
            self.writer.add_scalar('Loss/vae_loss', vae_loss, global_step=self.num_critic_update_iteration)

            mu, log_var = self.vae.encode(state)
            state = self.vae.reparameterization(mu, log_var)

            mu, log_var = self.vae.encode(next_state)
            next_state = self.vae.reparameterization(mu, log_var)


            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state.detach(), action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state.detach(), action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)

            # Delayed policy updates:
            if num_iteration % 5 == 0:
                print("-----------------------Update Actor-------------------")
                with torch.no_grad():
                    actions = self.actor(state)
                actions.requires_grad = True

                # print(self._actor.get_params())
                Q = self.critic_1(state, actions)
                pg_loss = torch.mean(torch.sum(Q, 1))

                self.critic_1.zero_grad()
                pg_loss.backward()
                delta_a = deepcopy(actions.grad.data)
                actions = self.actor(Variable(state))
                delta_a[:] = self._invert_gradients(delta_a, actions, inplace=True)
                out = -torch.mul(delta_a, actions)
                self.actor.zero_grad()
                out.backward(torch.ones(out.shape).to(device))
                # if self.clip_grad > 0:
                #     torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self.clip_grad)
                self.actor_optimizer.step()



                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1


def main():
    agent = TD3(state_dim, action_dim, hidden_dim, latent_dim, max_action)

    for i_episode in range(6000):
        state = env.reset()

        Reward = 0
        Rate = 0
        Energy = 0
        for t in range(100):

            if i_episode < 30:
                action = np.zeros(4)
                for n in range(4):
                    action[n] = random.uniform(-1, 1)

            else:
                action = agent.select_action(state)

            action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
            action = action.clip(-1,1)
            next_state, reward, done, info, rate, energy = env.step(action,i_episode,t,1)

            Reward += reward
            Rate += rate
            Energy += energy

            if t == 99:
                done = 1
            else:
                done = 0

            agent.memory.push((state, next_state, action, reward, done))

            if i_episode >= 10:
                agent.update(t)

            state = next_state


        print("Episode:", i_episode, "Reward:", Reward / 100, "Rate:", Rate / 100, "Power:", Energy/100)



if __name__ == '__main__':
    main()