#!/usr/bin/env python3

import os

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from spinup.algos.pytorch.dqn.core import DQN

mpl.use("TKAgg")


def main(
    seed=0,
    seed_weights_init=0,
    exp_name="dqn",
    num_steps=5000,
    checkpoint=1000,
    T=1,
    batch_size=64,
    tau=1e-3,
    lr=5e-4,
    gamma=0.99,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    C=int(1e5),
    to_log=False,
    to_plot=False,
    device="cpu",
    hidden_sizes=[64, 64]
):

    env = gym.make("LunarLander-v2")
    env.seed(seed)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    # Learner setup block
    torch.manual_seed(seed)
    agent = DQN(
        o_dim, a_dim,
        seed,
        seed_weights_init,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        lr=lr,
        buffer_size=C,
        device=device,
        hidden_sizes=hidden_sizes)

    # Experiment block starts
    ret = 0
    rets = []
    avgrets = []
    o = env.reset()

    eps = eps_start
    for steps in tqdm(range(num_steps)):

        # Select an action
        a = agent.act(o, eps=eps)

        # Observe
        op, r, done, _ = env.step(a.item())

        # Learn
        agent.memory.add(o, a, r, op, done)

        # Time to learn
        if (steps + 1) % T == 0:
            if len(agent.memory) > batch_size:
                experiences = agent.memory.sample()
                agent.learn(experiences)

        if (steps + 1) % max_t == 0:
            eps = max(eps_end, eps_decay*eps)

        o = op

        # Log
        ret += r
        if done:
            rets.append(ret)
            ret = 0
            o = env.reset()

        if (steps + 1) % checkpoint == 0:
            avgrets.append(np.mean(rets))

            if to_log:
                print(f"{avgrets[-1]:.5f}")
            rets = []

            if to_plot:
                plt.clf()
                plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
                plt.pause(0.001)

    data = np.zeros((2, len(avgrets)))
    data[0] = range(checkpoint, num_steps + 1, checkpoint)
    data[1] = avgrets
    data_dir = os.path.join("data", exp_name)
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(os.path.join(data_dir, str(seed) + ".txt"), data)

    if to_plot:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--seed_weights_init', '-swi', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--checkpoint', type=int, default=10000)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max_t', type=float, default=1000)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.995)
    parser.add_argument('--C', type=int, default=int(1e5))
    parser.add_argument('--to_log', type=bool, default=False)
    parser.add_argument('--to_plot', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default="dqn")
    parser.add_argument('--hidden_sizes', nargs="+", default=[64, 64])
    args = parser.parse_args()

    main(
        seed=args.seed,
        seed_weights_init=args.seed_weights_init,
        exp_name=args.exp_name,
        num_steps=args.num_steps,
        checkpoint=args.checkpoint,
        T=args.T,
        batch_size=args.batch_size,
        tau=args.tau,
        lr=args.lr,
        gamma=args.gamma,
        max_t=args.max_t,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        C=args.C,
        to_log=args.to_log,
        to_plot=args.to_plot,
        device=args.device,
        hidden_sizes=args.hidden_sizes
    )



