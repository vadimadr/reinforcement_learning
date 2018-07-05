import torch

from gym.spaces import Discrete, Box

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent(object):
    """Base class for all agents"""

    def get_action(self, state):
        """Returns policy action for a given state"""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Updates policy parameters given generated sessions"""
        raise NotImplementedError


def obesrvation_shape(observation_space):
    if isinstance(observation_space, Discrete):
        return (observation_space.n,)

    elif isinstance(observation_space, Box):
        return observation_space.shape

    raise NotImplementedError
