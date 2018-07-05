import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from common import obesrvation_shape, device, Agent
from utils import get_cumulative_rewards, generate_session


class ActorCritic(Agent):
    def __init__(self, observation_space, action_space, args):
        self.model = MlpPolicy(observation_space, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.lr)
        self.gamma = args.gamma
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

    def save(self, path):
        torch.save({
            'state_dict': self.model.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        state_dict = checkpoint.get('state_dict', checkpoint)
        self.model.load_state_dict(state_dict)

    def update(self, states, actions, rewards):
        states = torch.tensor(states).to(device, torch.float)
        actions = torch.tensor(actions).to(device, torch.float)
        cumulative_returns = torch.tensor(get_cumulative_rewards(rewards, self.gamma)).to(device)

        # predict logits, probas and log-probas using an agent.
        logits, values = self.model(states)
        probas = F.softmax(logits, dim=1)
        logprobas = F.log_softmax(logits, dim=1)

        # select log-probabilities for chosen actions, log pi(a_i|s_i)
        logprobas_for_actions = torch.sum(logprobas * to_one_hot(actions).to(torch.float), dim=1)

        # REINFORCE objective function
        self.optimizer.zero_grad()
        rewards_with_baseline = cumulative_returns - values.squeeze(1)
        policy_loss = torch.mean(-logprobas_for_actions * rewards_with_baseline)
        value_loss = F.smooth_l1_loss(values, cumulative_returns.unsqueeze(1))
        entropy = torch.mean(-torch.sum(logprobas * probas, dim=1))
        loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy
        loss.backward()
        self.optimizer.step()

        # technical: return session rewards to print them later
        return np.sum(rewards)

    def get_action(self, state):
        with torch.no_grad():
            logits, _ = self.model(torch.tensor(state.reshape(1, -1)).to(device, torch.float))
            dist = Categorical(F.softmax(logits, dim=1))
            return dist.sample().item()


class MlpPolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(MlpPolicy, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(*obesrvation_shape(observation_space), 50),
            nn.ReLU(),
            # nn.Linear(50, 50),
            # nn.ReLU(),
        )
        self.logits = nn.Linear(50, action_space.n)
        self.state_values = nn.Linear(50, 1)

    def forward(self, x):
        x = self.model(x)
        return self.logits(x), self.state_values(x)


def to_one_hot(y, n_dims=None):
    """ Take an integer vector (tensor of variable) and convert it to 1-hot matrix. """
    y_tensor = y.to(torch.long).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = y_tensor.new_zeros(y_tensor.size(0), n_dims).scatter_(1, y_tensor, 1)
    return y_one_hot


def train(agent, env, args):
    total_rewars = []
    for i in range(args.num_steps):
        states, actions, rewards = generate_session(agent, env, args.max_episode_length)
        total_reward = agent.update(states, actions, rewards)
        total_rewars.append(total_reward)

        current_mean_reward = np.mean(total_rewars[-100:])
        if i % 100 == 0:
            print("Iteration: %i, Mean reward:%.3f" % (i // 100, current_mean_reward))
            if current_mean_reward > 200:
                return
