import gym

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def make_env(env_name, seed=None, num_envs=1, num_processes=1):
    envs = [_make_env(env_name, seed + i) for i in range(num_envs)]
    if num_processes > 1:
        env = SubprocVecEnv(envs)
    else:
        env = DummyVecEnv(envs)

    return env


def _make_env(env_name, seed):
    def wrapped():
        env = gym.make(env_name)
        env.seed(seed)

        return env

    return wrapped
