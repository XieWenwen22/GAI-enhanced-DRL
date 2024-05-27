import argparse
import math
import random
from typing import Type

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from diffusion import Diffusion
from model1 import MLP, DoubleCritic
from policy1 import DiffusionOPT
import seaborn as sns

from Environment_relay_simple_state import Environment

# writer = SummaryWriter("Loss")
def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-noise", type=float, default=0.01) # default=0.01
    parser.add_argument('--algorithm', type=str, default='diffusion_opt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1e6)#1e6
    parser.add_argument('-e', '--epoch', type=int, default=1e6)# 1000
    parser.add_argument('--step-per-epoch', type=int, default=1)# 100
    parser.add_argument('--step-per-collect', type=int, default=1)#1000
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0)
    # parser.add_argument(
    #     '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--device', type=str, default='cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action='store_true', default=False)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--note', type=str, default='')

    # for diffusion
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    # adjust
    parser.add_argument('-t', '--n-timesteps', type=int, default=5)  # for diffusion chain 3 & 8 & 12
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])

    # whether the expert action is availiable
    parser.add_argument('--expert-coef', default=True)

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.6)#0.6
    parser.add_argument('--prior-beta', type=float, default=0.4)#0.4

    # Parse arguments and return them
    args = parser.parse_known_args()[0]
    return args

# def location(UAV_location):


def main(args=get_args()):
    # create environments
    env = Environment()
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.max_action = 1.

    args.exploration_noise = args.exploration_noise * args.max_action

    # create actor
    actor_net = MLP(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    )
    # Actor is a Diffusion model
    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=actor_net,
        max_action=args.max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps
    ).to(args.device)
    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )

    # Create critic
    critic1 = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    critic_optim1 = torch.optim.Adam(
        critic1.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    # Create critic
    critic2 = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    critic_optim2 = torch.optim.Adam(
        critic2.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    policy = DiffusionOPT(
        args.state_shape,
        actor,
        actor_optim,
        args.action_shape,
        critic1,
        critic_optim1,
        critic2,
        critic_optim2,
        # dist,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        expert_coef=args.expert_coef,
        action_space=env.action_space,
    )

    total_steps = 0
    start_epsilon = 1
    end_epsilon = 0
    epsilon_steps = 5
    writer = SummaryWriter("GDMTD3")
    for i_episode in range(6000):
        # print("-----------------------------------------------------------------------------")
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        epsilon = end_epsilon + (start_epsilon - end_epsilon) * \
                       math.exp(-1. * i_episode / 30)

        total_critic_loss = 0
        total_actor_loss = 0
        time = 0

        Reward = 0
        Rate = 0
        Power = 0

        for t in range(100):

            if random.random() < epsilon:
                action = np.zeros(4)
                for n in range(4):
                    action[n] = random.uniform(-1,1)

            else:
                action = policy.select_action(state,i_episode)


            next_state, reward, info, done, rate, power = env.step(action,i_episode,t,1)
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            if t == 99:
                done = 1
            else:
                done = 0


            Reward += reward
            Rate += rate
            Power += power

            # adding in memory
            policy.add_samples(state, action, reward, next_state, terminal=done)
            state = next_state

            # train the DDPG agent if needed
            if total_steps > policy.batch_size*5:
                # print("Update")
                # print(actor_net.get_params())
                loss = policy.learn(t)
                # print(actor_net.get_params())
                critic_loss = loss.get("loss/critic")
                actor_loss = loss.get("overall_loss")
                total_critic_loss += critic_loss
                total_actor_loss += actor_loss
                # print(loss.get("loss/critic"),loss.get("overall_loss"))


            total_steps += 1
            time += 1

            # writer.add_scalar("Reward",Reward/100,i_episode*100+t)

        print("Episode:",i_episode,"Reward:",Reward/100, "Rate:",Rate/100, "Power",Power/100)
        print("------------------------------------------------------------------------------------------------")



main()







