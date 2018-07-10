import gym

from multi_env import MultiEnvWrapper


def make_env(env_name, seed=0, num_envs=1, num_processes=1):
    envs = [_make_env(env_name, seed + i) for i in range(num_envs)]

    env = MultiEnvWrapper(envs, num_workers=num_processes)

    return env


def _make_env(env_name, seed):
    def wrapped():
        env = gym.make(env_name)
        env.seed(seed)

        return env

    return wrapped
