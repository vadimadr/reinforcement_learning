import argparse

from a2c.actor_critic import train, ActorCritic
from utils import make_fun
from env import make_env
from logger import init_logging


def get_argparser():
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=50,
                        help='value loss coefficient')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--num-steps', type=int, default=int(1e4),
                        help='number of forward steps')
    parser.add_argument('--episode-length', type=int, default=int(256),
                        help='maximum length of an episode')
    parser.add_argument('--env', default='CartPole-v0',
                        help='environment to train on')
    parser.add_argument('-j', '--num-processes', default=1, type=int,
                        help='Number of environment processes')
    parser.add_argument('--num-envs', default=1, type=int,
                        help='Number of environment')

    return parser


def main():
    args = get_argparser().parse_args()
    init_logging('logs')

    env = make_env(args.env, args.seed, num_envs=args.num_envs, num_processes=args.num_processes)
    agent = ActorCritic(env.observation_space, env.action_space, args)
    agent.save("checkpoint.pth")

    train(agent, env, args)
    make_fun(agent, env, render=True)


if __name__ == '__main__':
    main()
