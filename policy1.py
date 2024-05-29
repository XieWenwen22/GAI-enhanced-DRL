import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union

from torch import tensor
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from Environment_relay_simple_state import Environment
from helpers import (
    Losses
)
from memory import Memory

env = Environment()

class DiffusionOPT():

    def __init__(
            self,
            state_dim,
            actor,
            actor_optim,
            action_dim,
            critic1,
            critic_optim1,
            critic2,
            critic_optim2,
            # dist_fn: Type[torch.distributions.Distribution],
            device,
            tau = 0.005,
            gamma = 0.9,
            reward_normalization = False,
            estimation_step = 1,
            lr_decay = False,
            lr_maxt = 1000,
            expert_coef = False,
            **kwargs
    ) -> None:
        super().__init__()
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        # Initialize actor network and optimizer if provided
        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor  # Actor network
            self._target_actor = deepcopy(actor)  # Target actor network for stable learning
            self._target_actor.eval()  # Set target actor to evaluation mode
            self._actor_optim: torch.optim.Optimizer = actor_optim  # Optimizer for the actor network
            self._action_dim = action_dim  # Dimensionality of the action space

        # Initialize critic network and optimizer if provided
        if critic1 is not None and critic_optim1 is not None:
            self._critic1: torch.nn.Module = critic1  # Critic network
            self._target_critic1 = deepcopy(critic1)  # Target critic network for stable learning
            self._critic_optim1: torch.optim.Optimizer = critic_optim1  # Optimizer for the critic network
            self._target_critic1.eval()  # Set target critic to evaluation mode

        if critic2 is not None and critic_optim2 is not None:
            self._critic2: torch.nn.Module = critic2  # Critic network
            self._target_critic2 = deepcopy(critic2)  # Target critic network for stable learning
            self._critic_optim2: torch.optim.Optimizer = critic_optim2  # Optimizer for the critic network
            self._target_critic2.eval()  # Set target critic to evaluation mode

        # If learning rate decay is applied, initialize learning rate schedulers for both actor and critic
        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler1 = CosineAnnealingLR(self._critic_optim1, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler2 = CosineAnnealingLR(self._critic_optim2, T_max=lr_maxt, eta_min=0.)

        self.memory = Memory(
            10**(6), env.observation_space.shape,
            env.action_space.shape, next_actions=False)

        self.UAV_measurements = np.zeros((500,500))

        # Initialize other parameters and configurations
        self._steps = 0
        self.batch_size = 128
        self._tau = tau  # Soft update coefficient for target networks
        self._gamma = gamma  # Discount factor for future rewards
        self._rew_norm = reward_normalization  # If true, normalize rewards
        self._n_step = estimation_step  # Steps for n-step return estimation
        self._lr_decay = lr_decay  # If true, apply learning rate decay
        self._expert_coef = expert_coef  # Coefficient for policy gradient loss
        self._device = device  # Device to run computations on
        self.clip_grad = 10

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

    def draw_map(self,state):
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        logits = self._actor(state).detach().numpy()
        # print("logits:",logits)
        # logits += np.clip(np.random.normal(0, 0.02), -0.02, 0.02)
        # logits = logits
        if logits[0][0] < -1:
            logits[0][0] = -1
        elif logits[0][0] > 1:
            logits[0][0] = 1

        return logits[0],0

    def select_action(self,state,i_episode):

        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        # print("state:",state)
        logits = self._actor(state).detach().numpy()
        # logits += np.clip(np.random.normal(0, 0.02), -0.02, 0.02)
        # logits = logits
        if logits[0][0] < -1:
            logits[0][0] = -1
        elif logits[0][0] > 1:
            logits[0][0] = 1

        return logits[0]

    def add_samples(self,state, action, reward, next_state, terminal):
        self._steps += 1
        self.memory.append(state, action, reward, next_state, terminal=terminal)



    def to_one_hot(
            self,
            data: np.ndarray,
            one_hot_dim: int
    ) -> np.ndarray:
        # Convert the provided data to one-hot representation
        batch_size = data.shape[0]
        one_hot_codes = np.eye(one_hot_dim)
        # print(data[1])
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim))
                       for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)


    def update_critic(self, experiences):
        # Compute the critic's loss and update its parameters
        states, actions, rewards, next_states, terminals = experiences
        next_actions = self._target_actor(next_states)

        target_Q1 = self._target_critic1(next_states, next_actions)
        target_Q2 = self._target_critic2(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + ((1 - terminals) * self._gamma * target_Q).detach()

        current_Q1 = self._critic1(states, actions)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self._critic_optim1.zero_grad()
        loss_Q1.backward()
        self._critic_optim1.step()

        # Optimize Critic 2:
        current_Q2 = self._critic2(states, actions)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self._critic_optim2.zero_grad()
        loss_Q2.backward()
        self._critic_optim2.step()

        return loss_Q1


    def update_policy(self, experience):
        # Compute the policy gradient loss
        states, actions, rewards, next_states, terminals = experience
        with torch.no_grad():
            actions = self._actor(states)
        actions.requires_grad = True

        # print(self._actor.get_params())
        Q = self._critic1(states,actions)
        pg_loss = torch.mean(torch.sum(Q,1))
        # pg_loss1 = copy.deepcopy(pg_loss)
        # pg_loss1 = -pg_loss1
        # if update: # If update flag is set, backpropagate the loss and perform a step of optimization

        self._critic1.zero_grad()
        pg_loss.backward()
        delta_a = deepcopy(actions.grad.data)
        actions = self._actor(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, actions, inplace=True)
        out = -torch.mul(delta_a, actions)
        self._actor.zero_grad()
        out.backward(torch.ones(out.shape).to(self._device))
        # if self.clip_grad > 0:
        #     torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self.clip_grad)
        self._actor_optim.step()

        # print(self._actor.get_params())
        return -pg_loss

    def soft_update(self, tgt, src, tau):
        """Softly update the parameters of target module towards the parameters \
        of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def update_targets(self):
        # Perform soft update on target actor and target critic. Soft update is a method of slowly blending
        # the regular and target network to provide more stable learning updates.
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic1, self._critic1, self._tau)
        self.soft_update(self._target_critic2, self._critic2, self._tau)

    def learn(self,actor_update_freq) :
        # Update critic network. The critic network is updated to minimize the mean square error loss
        # between the Q-value prediction (current_q1) and the target Q-value (target_q).
        states, actions, rewards, next_states, terminals = self.memory.sample(self.batch_size)
        states = torch.from_numpy(states).to(self._device)
        actions = torch.from_numpy(actions).to(self._device)  # make sure to separate actions and parameters
        rewards = torch.from_numpy(rewards).squeeze(0).to(self._device)
        next_states = torch.from_numpy(next_states).to(self._device)
        terminals = torch.from_numpy(terminals).squeeze(0).to(self._device)
        experinces = states, actions, rewards, next_states, terminals

        critic_loss = self.update_critic(experinces)
        # Update actor network. Here, we first calculate the policy gradient (pg_loss) and
        # behavior cloning loss (bc_loss) but we do not update the actor network yet.
        #
        if actor_update_freq % 5 == 0:
            pg_loss = self.update_policy(experinces)
        else:
            pg_loss = tensor(0)

        self.update_targets()
        return {
            'loss/critic': critic_loss.item(),  # Returns the critic loss as part of the results
            'overall_loss': pg_loss.item()  # Returns the overall loss as part of the results
        }