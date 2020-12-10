import argparse
import gym

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--seed_weight_init', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--steps_per_epoch', type=int)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_fname', type=str)
parser.add_argument('--exp_name', type=str)
args = parser.parse_args()

if args.algo == 'ppo':
	from spinup import ppo_pytorch as algo
elif args.algo == 'vpg':
	from spinup import vpg_pytorch as algo

print('using algo', args.algo)
# print('steps_per_epoch', steps_per_epoch)
# print('epochs', epochs)
# print('se', epochs)

env_fn = lambda : gym.make('LunarLander-v2')
logger_kwargs = dict(output_dir=args.output_dir, exp_name=args.exp_name, output_fname=args.output_fname)
algo(
	env_fn=env_fn, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
	seed=args.seed, seed_weight_init=args.seed_weight_init,
	logger_kwargs=logger_kwargs
)

