import time
import gym
import numpy as np
import torch
from tqdm import tqdm
from spinup.utils.logx import EpochLogger
from spinup.algos.pytorch.dqn.core import DQN


def dqn(
    env_fn,
    ac_kwargs=dict(),
    seed=0,
    seed_weight_init=None,
    steps_per_epoch=4000, epochs=100,
    replay_size=1000000, gamma=0.99,
    q_lr=5e-4, batch_size=64,
    update_after=100, update_every=1,
    max_ep_len=1000,
    tau=1e-3,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    logger_kwargs=dict(), save_freq=1):


    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    env.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape
    if all(act_dim):
        act_dim = env.action_space.n

    hidden_sizes = ac_kwargs["hidden_sizes"]
    agent = DQN(
        obs_dim, act_dim,
        seed,
        seed_weight_init,
        buffer_size=replay_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        lr=q_lr,
        hidden_sizes=hidden_sizes)


    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    eps = eps_start
    for t in range(total_steps):

        # Select an action
        a = agent.act(o, eps=eps)

        # Observe
        op, r, d, _ = env.step(a.item())
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Learn
        agent.memory.add(o, a, r, op, d)

        o = op

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Time to learn
        if (t + 1) % update_every == 0:
            if len(agent.memory) > update_after:
                experiences = agent.memory.sample()
                agent.learn(experiences)

        if (t + 1) % max_ep_len == 0:
            eps = max(eps_end, eps_decay*eps)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0, required=True)
    parser.add_argument('--seed_weight_init', '-swi', type=int, default=0, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='dqn')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    dqn(
        lambda: gym.make(args.env),
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed,
        seed_weight_init=args.seed_weight_init,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs
    )



