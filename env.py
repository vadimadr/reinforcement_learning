import gym


def make_env(env_name, seed=None):
    env = gym.make(env_name).env

    return env
